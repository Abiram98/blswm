"""Normalize execution logs and map overlap and available host measurements."""
import csv
import hashlib
import heapq
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
from wrk_gen.queries import fingerprint
from .io import output_directory


def _times(path):
    start=end=None
    for line in path.read_text().splitlines():
        if line.startswith(('start:', 'Workload start:')):
            value=float(line.split(':',1)[1].strip())
            if start is None:start=value
            else:end=value
        elif line.startswith(('end:', 'Workload end:')):end=float(line.split(':',1)[1].strip())
    return start,end


def discover(paths):
    result=[]
    for value in paths:
        path=Path(value).expanduser().resolve()
        if path.is_file() and path.name=='run.json':path=path.parent
        if not path.is_dir():raise ValueError(f'Input is not a run or directory: {path}')
        if (path/'run.json').is_file() or (path/'wkl_time').is_dir() or (path/'workload_time').is_dir():
            result.append(path);continue
        for run in path.rglob('run.json'):result.append(run.parent)
        for name in ('wkl_time','workload_time'):
            for folder in path.rglob(name):
                if folder.is_dir():result.append(folder.parent)
    result=list(dict.fromkeys(result))
    if not result:raise ValueError('No execution runs found')
    return result


def read_run(path, time_base='epoch', resource_interval=.1):
    path=Path(path).resolve(); hashes={}
    if (path/'run.json').is_file():
        source=path/'run.json'; hashes[str(source)]=fingerprint(source)
        run=json.loads(source.read_text());records=run['records']
        if not records:raise ValueError(f'No records in {path}')
        result=[]
        for row in records:
            result.append(dict(query_id=str(row['query_id']),occurrence_id=str(row['occurrence_id']),
                arrival_s=row.get('arrival_s'),start_s=row.get('admission_s'),end_s=row.get('completion_s'),
                wait_s=row.get('wait_s'),execution_s=row.get('execution_s'),latency_s=row.get('latency_s'),
                status=row.get('status','unknown'),error=row.get('error'),slot=row.get('slot')))
        start=0.;end=max((r['end_s'] for r in result if r['end_s'] is not None),default=0.)
        meta=dict(source_format='run.json',complete=run.get('complete'),policy=run.get('configuration',{}).get('policy') or records[0].get('policy'),
                  mpl=run.get('mpl'),run_id=run.get('run_id'),clock='relative_monotonic')
    else:
        folder=path/('wkl_time' if (path/'wkl_time').is_dir() else 'workload_time')
        startfile=folder/'wkl_start.log';endfile=folder/'wkl_end.log'
        if not startfile.is_file() or not endfile.is_file():raise ValueError(f'Missing workload start/end logs: {folder}')
        start,_=_times(startfile);_,end=_times(endfile)
        if start is None or end is None or end<start:raise ValueError(f'Invalid workload timing: {folder}')
        hashes.update({str(p):fingerprint(p) for p in (startfile,endfile)})
        result=[]
        for file in sorted(folder.glob('*.log')):
            if file.name in ('wkl_start.log','wkl_end.log'):continue
            hashes[str(file)]=fingerprint(file);a,b=_times(file)
            if a is None or b is None:raise ValueError(f'Missing query start/end: {file}')
            raw=file.read_text();match=re.search(r'^active_key:\s*\d+:(.+)$',raw,re.M)
            identity=match.group(1).strip() if match else re.sub(r'_\d{9,}(?:_(?:fast|slow|Pool_[AB]))?$','',file.stem)
            status=re.search(r'^terminal_status:\s*(.+)$',raw,re.M)
            status=status.group(1).strip() if status else 'unknown'
            if status in ('completed','success'):status='success'
            waitfile=path/'wait_times'/(file.stem+'.wait');wait=None;arrival=None
            if waitfile.is_file():
                hashes[str(waitfile)]=fingerprint(waitfile)
                try:
                    obj=json.loads(waitfile.read_text());wait=obj.get('wait_time');arrival=obj.get('expected_arrival')
                except (ValueError,AttributeError):
                    values=dict(line.split(':',1) for line in waitfile.read_text().splitlines() if ':' in line)
                    wait=float(values['wait_time']) if 'wait_time' in values else None
                    arrival=float(values['expected_arrival']) if 'expected_arrival' in values else None
            result.append(dict(query_id=identity,occurrence_id=identity,arrival_s=float(arrival)-start if arrival is not None else None,
                start_s=a-start,end_s=b-start,wait_s=float(wait) if wait is not None else None,execution_s=b-a,
                latency_s=b-float(arrival) if arrival is not None else None,status=status,error=None,slot=None))
        meta=dict(source_format='wkl_time',complete=None,policy=None,mpl=None,run_id=path.name,clock='epoch_or_monotonic_relative')
    if len({r['occurrence_id'] for r in result})!=len(result):raise ValueError(f'Duplicate occurrence IDs: {path}')
    for r in result:
        for name in ('arrival_s','start_s','end_s','wait_s','execution_s','latency_s'):
            value=r[name]
            if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)):
                raise ValueError(f'Nonfinite {name} in {path}')
        if r['start_s'] is not None and r['end_s'] is not None and r['end_s']<r['start_s']:
            raise ValueError(f'Reversed query interval in {path}')
    valid=[r for r in result if r['start_s'] is not None and r['end_s'] is not None]
    overlaps={r['occurrence_id']:[] for r in valid}
    active={};heap=[]
    for row in sorted(valid,key=lambda r:(r['start_s'],r['occurrence_id'])):
        while heap and heap[0][0]<=row['start_s']:
            _,key=heapq.heappop(heap);active.pop(key,None)
        if row['end_s']<=row['start_s']:continue
        key=row['occurrence_id']
        for other in active:
            overlaps[key].append(other);overlaps[other].append(key)
        active[key]=row;heapq.heappush(heap,(row['end_s'],key))
    for row in result:
        found=overlaps.get(row['occurrence_id'])
        row['overlap_count']=len(found) if found is not None else None
        row['overlapping_queries']=json.dumps(found) if found is not None else None
    resource=path/'res'/'resource_trace.csv'
    if resource.is_file():
        hashes[str(resource)]=fingerprint(resource)
        if meta['source_format']=='run.json':
            meta['resource_mapping']='unavailable: relative query times lack an epoch anchor'
        else:
            data=pd.read_csv(resource)
            if time_base=='epoch':
                if '#Date' not in data or 'Time' not in data:raise ValueError(f'Missing resource date/time columns: {resource}')
                raw_times=data['#Date'].astype(str)+' '+data['Time'].astype(str)
                times=pd.to_datetime(raw_times,utc=True,errors='coerce')
                if times.isna().any():
                    times=raw_times.map(lambda value: pd.to_datetime(value,utc=True,errors='coerce'))
                if times.isna().any():raise ValueError(f'Invalid resource timestamps: {resource}')
                data['_time']=times.astype('int64')/1e9
                origin=start
            else:
                data['_time']=np.arange(len(data))*resource_interval
                origin=0.
            resource_columns=['[CPU]User%','[CPU]Sys%','[CPU]Totl%','[MEM]Tot','[MEM]Used','[MEM]Buf','[MEM]Cached']
            for col in resource_columns:
                if col in data:data[col]=pd.to_numeric(data[col],errors='coerce')
            for r in valid:
                window=data[(data['_time']>=r['start_s']+origin)&(data['_time']<=r['end_s']+origin)]
                measurements={col:[float(v) if math.isfinite(v) else None for v in window[col]] if col in window else [] for col in resource_columns}
                r['resource_measurements']=json.dumps(measurements,allow_nan=False)
                for col,key in [('[CPU]Totl%','host_cpu_total_pct'),('[MEM]Used','host_memory_used')]:
                    if col in window:
                        x=pd.to_numeric(window[col],errors='coerce').dropna()
                        r[key+'_mean']=float(x.mean()) if len(x) else None
                        r[key+'_peak']=float(x.max()) if len(x) else None
                r['resource_samples']=len(window)
            meta['resource_mapping']=time_base
    else:meta['resource_mapping']='unavailable: no resource_trace.csv'
    for filename,digest in hashes.items():
        if fingerprint(filename)!=digest:raise ValueError(f'Input changed: {filename}')
    return result,dict(meta,source=str(path),workload_duration_s=end-start,input_sha256=hashes,
                       records=len(result),successful=sum(r['status']=='success' for r in result))


def prepare(paths, output, *, time_base='epoch', resource_interval=.1):
    if time_base not in ('epoch','perf_counter') or not math.isfinite(resource_interval) or resource_interval<=0:
        raise ValueError('Invalid resource time base or interval')
    paths=discover(paths); loaded=[read_run(path,time_base,resource_interval) for path in paths]
    destination=output_directory(output,paths)
    entries=[]
    for path,(records,meta) in zip(paths,loaded):
        name=hashlib.sha256(str(path).encode()).hexdigest()[:12]; folder=destination/name;folder.mkdir()
        fields=list(dict.fromkeys(key for row in records for key in row))
        with (folder/'executions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
        (folder/'summary.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        entries.append(dict(source=str(path),folder=name,records=len(records),complete=meta['complete']))
    (destination/'index.json').write_text(json.dumps(dict(schema_version=1,runs=entries),indent=2)+'\n')
    return dict(output=str(destination),runs=len(entries),records=sum(e['records'] for e in entries))
