"""Sequential endpoint executions paired with database-host resource recordings."""
from bisect import bisect_right
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .groups import group_paths
from .monitor import Monitor
from .queries import fingerprint, load_queries, new_directory, write_csv, write_json


def execute(endpoint, text, timeout):
    start, epoch = time.monotonic(), time.time()
    result = dict(start_s=start, start_epoch=epoch, status='success', error=None,
                  http_status=None, response_bytes=0, response_headers={})
    digest = hashlib.sha256()
    try:
        data = urlencode(dict(query=text, timeout=str(int(timeout*1000)))).encode()
        request = Request(endpoint, data=data, headers={'Accept':'application/sparql-results+json'})
        with urlopen(request, timeout=timeout+5) as response:
            result.update(http_status=response.status, response_headers=dict(response.headers.items()))
            while block := response.read(65536):
                result['response_bytes'] += len(block)
                digest.update(block)
            sql_state = response.headers.get('X-SQL-State')
            if sql_state and sql_state != '00000':
                result.update(status='endpoint_error',error=f'{sql_state}: {response.headers.get("X-SQL-Message", "")}')
    except HTTPError as exc:
        result.update(status='endpoint_error',http_status=exc.code,error=f'{exc}: {exc.read(2048).decode(errors="replace")}')
    except Exception as exc:
        result.update(status='transport_error',error=f'{type(exc).__name__}: {exc}')
    result.update(end_s=time.monotonic(),end_epoch=time.time(),response_sha256=digest.hexdigest())
    result['runtime_seconds'] = result['end_s']-start
    return result


def collectl_playback(folder, pid):
    raw = sorted(folder.glob('*.raw*'))
    if not raw:
        raise RuntimeError('No collectl raw files were retained')
    target = folder/'playback.txt'
    with target.open('w') as handle:
        for path in raw:
            command = ['collectl','-p',str(path),'-sZ','-P','-oTm']
            result = subprocess.run(command,stdout=handle,stderr=subprocess.PIPE,text=True,
                                    env=dict(os.environ,TZ='UTC',LC_ALL='C'),timeout=120)
            if result.returncode:
                raise RuntimeError(f'collectl playback failed: {result.stderr[-1000:]}')
    rows, header = [], None
    with target.open() as handle:
        for line in handle:
            if line.startswith('#Date Time PID '):
                header = line.lstrip('#').split()
            elif header and line and not line.startswith('#'):
                values = line.split(None,len(header)-1)
                if len(values)!=len(header) or not values[0].isdigit():
                    continue
                value = dict(zip(header,values))
                if int(value['PID'])==pid:
                    timestamp=datetime.strptime(value['Date']+' '+value['Time'],
                                                '%Y%m%d %H:%M:%S.%f' if '.' in value['Time'] else '%Y%m%d %H:%M:%S')
                    rows.append(dict(epoch=timestamp.replace(tzinfo=timezone.utc).timestamp(),pid=pid,
                                     cpu_pct=float(value['PCT']),rss_bytes=int(value['VmRSS'])*1024))
    if not rows:
        raise RuntimeError('collectl playback contains no records for the database PID')
    write_csv(folder/'collectl_samples.csv',rows)
    return len(rows)


def cpu_measurements(samples, record, offset, window):
    start, end = record['start_s'], record['end_s']
    times = [row['monotonic']+offset for row in samples]
    cpu = [row['cpu_seconds'] for row in samples]
    if any(b<=a for a,b in zip(times,times[1:])) or any(b<a for a,b in zip(cpu,cpu[1:])):
        raise ValueError('Nonmonotonic database process counters')
    peaks, rss = [], []
    for index,right in enumerate(times):
        if start<=right<=end:
            rss.append(samples[index]['rss_bytes'])
        left = right-window
        if left<start or right>end:
            continue
        lower = bisect_right(times,left)-1
        if lower<0 or lower>=index:
            continue
        proportion = (left-times[lower])/(times[lower+1]-times[lower])
        baseline = cpu[lower]+proportion*(cpu[lower+1]-cpu[lower])
        peaks.append(max(0.,(cpu[index]-baseline)/window*100))
    before, after = record['process_before'], record['process_after']
    duration = after['monotonic']-before['monotonic']
    return dict(peak_cpu_pct=max(peaks) if peaks else None, cpu_windows=len(peaks),
                cpu_seconds=after['cpu_seconds']-before['cpu_seconds'],
                mean_cpu_pct=(after['cpu_seconds']-before['cpu_seconds'])/duration*100 if duration>0 else None,
                peak_rss_bytes=max(rss) if rss else None, cpu_window_seconds=window)


def read_samples(path):
    with path.open(newline='') as handle:
        return [{key:float(value) for key,value in row.items()} for row in csv.DictReader(handle)]


def profile_groups(inputs, output, endpoint, *, ssh_host=None, local=False, ssh_user='root', ssh_port=22,
                   ssh_identity_file=None, ssh_password_env='WRK_GEN_SSH_PASSWORD', pid=None,
                   query_timeout=30., repetitions=1, cpu_interval=.001, collectl_interval=.01,
                   cpu_window=.01, query_gap=.05):
    if repetitions<1 or any(not math.isfinite(v) or v<=0 for v in (query_timeout,cpu_interval,collectl_interval,cpu_window)):
        raise ValueError('Timeouts, repetitions and sampling intervals must be positive')
    if not math.isfinite(query_gap) or query_gap<0 or cpu_interval>cpu_window:
        raise ValueError('query-gap must be nonnegative and cpu-interval cannot exceed cpu-window')
    parsed = urlparse(endpoint)
    if parsed.scheme not in ('http','https') or not parsed.hostname:
        raise ValueError('Provide an HTTP(S) SPARQL endpoint')
    paths = group_paths(inputs)
    groups = [(path,load_queries(path)) for path in paths]
    identities = {}
    for path, queries in groups:
        for query in queries:
            key=query['query_id']
            if key in identities:
                raise ValueError('Query groups must have distinct query IDs')
            identities[key]=query
    hashes = {str(path/'ql.csv'):fingerprint(path/'ql.csv') for path in paths}
    destination = new_directory(output,paths)
    manifest = dict(schema_version=1,status='running',endpoint=endpoint,queries=len(identities),repetitions=repetitions,
                    query_timeout_seconds=query_timeout,cpu_interval_seconds=cpu_interval,
                    collectl_interval_seconds=collectl_interval,cpu_window_seconds=cpu_window,
                    cpu_units='percent; 100 percent equals one CPU core',
                    cpu_scope='whole database process, including background activity',
                    execution_mode='sequential; one query at a time',input_sha256=hashes)
    write_json(destination/'manifest.json',manifest)
    executions = []
    try:
        host = None if local else ssh_host or parsed.hostname
        with Monitor(host,ssh_user,ssh_port,pid,ssh_identity_file,ssh_password_env) as monitor:
            if cpu_window<monitor.info['counter_resolution_s']:
                raise ValueError('CPU window is smaller than the database counter resolution')
            manifest.update(database=monitor.info,clock_calibration=monitor.calibration,ssh_host=host)
            write_json(destination/'manifest.json',manifest)
            for index,(path,queries) in enumerate(groups,1):
                folder = destination/f'group_{index:04}'
                folder.mkdir()
                write_json(folder/'input.json',dict(path=str(path),sha256=hashes[str(path/'ql.csv')]))
                recording = monitor.start(cpu_interval,collectl_interval)
                write_json(folder/'collector.json',recording)
                group_records = []
                try:
                    time.sleep(max(.05,collectl_interval*2))
                    for query in queries:
                        for repetition in range(1,repetitions+1):
                            before = monitor.rpc('sample')
                            result = execute(endpoint,query['query_text'],query_timeout)
                            after = monitor.rpc('sample')
                            row = dict(query_id=query['query_id'],source_query_id=query['source_query_id'],
                                       query_text=query['query_text'],repetition=repetition,
                                       process_before=before,process_after=after,**result)
                            group_records.append(row)
                            with (folder/'executions.jsonl').open('a') as handle:
                                handle.write(json.dumps(row,allow_nan=False)+'\n')
                            print(json.dumps(dict(group=index,query_id=query['query_id'],repetition=repetition,
                                                  status=row['status'],runtime_seconds=row['runtime_seconds'])),flush=True)
                            if result['status']!='success':
                                raise RuntimeError('Query did not complete successfully; stopped to avoid overlapping an uncertain execution')
                            if query_gap:
                                time.sleep(query_gap)
                finally:
                    files = monitor.stop(folder/'collectl')
                    write_json(folder/'recordings.json',files)
                    sample_rows = read_samples(folder/'collectl/process_samples.csv')
                    normalized_count = collectl_playback(folder/'collectl',monitor.info['pid'])
                    for row in group_records:
                        row.update(cpu_measurements(sample_rows,row,monitor.calibration['offset'],cpu_window))
                    (folder/'executions.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in group_records))
                    executions.extend(group_records)
                    write_json(folder/'summary.json',dict(queries=len(queries),executions=len(group_records),collectl_process_records=normalized_count))
        catalog = []
        by_query = defaultdict(list)
        for row in executions:
            by_query[row['query_id']].append(row)
        for key,query in identities.items():
            rows=by_query[key]
            successful=[row for row in rows if row['status']=='success']
            if len(successful)!=repetitions:
                raise RuntimeError('Incomplete profiling repetitions')
            runtime=sum(row['runtime_seconds'] for row in successful)/len(successful)
            peaks=[row['peak_cpu_pct'] for row in successful if row['peak_cpu_pct'] is not None]
            peak=max(peaks) if peaks else None
            label=classify(runtime,peak)
            label={'HM':'H_M','HL':'H_L','L100':'V100','L':'L10'}.get(label,label) or 'unknown'
            catalog.append(dict(query_id=key,query_text=query['query_text'],source_query_id=query['source_query_id'],
                                runtime_seconds=runtime,isolated_runtime_s=runtime,peak_cpu_pct=peak,
                                mean_cpu_pct=sum(row['mean_cpu_pct'] for row in successful)/len(successful),
                                peak_rss_bytes=max((row['peak_rss_bytes'] or 0) for row in successful),
                                query_type=label,status='success',repetitions=repetitions,
                                cpu_window_seconds=cpu_window,query_sha256=query['query_sha256']))
        if any(fingerprint(path)!=digest for path,digest in hashes.items()):
            raise RuntimeError('Query group inputs changed during profiling')
        write_csv(destination/'catalog.csv',catalog)
        manifest.update(status='complete',executions=len(executions),catalog='catalog.csv')
        write_json(destination/'manifest.json',manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',executions=len(executions))
        write_json(destination/'manifest.json',manifest)
        raise


def classify(runtime, cpu):
    t = float(runtime)
    if not math.isfinite(t) or t < 0:
        return None
    if t < 1:
        return 'S'
    try:
        c = float(cpu)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c) or c < 100:
        return None
    if t < 10:
        return 'HM' if c > 200 else 'M'
    if t <= 120:
        return 'HL' if c > 200 else 'L100' if 100 <= t <= 110 else 'L'
    return None
