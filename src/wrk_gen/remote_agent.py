"""Standalone database-host collector. stdin/stdout carry JSON RPC only."""
import base64
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time


class Agent:
    def __init__(self):
        self.directory = self.collector = self.thread = self.log = None
        self.done = threading.Event()
        self.failure = None
        self.pid = None

    def point(self):
        fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()
        identity = int(fields[19])
        if identity != self.identity:
            raise RuntimeError('Database PID was reused')
        cpu = (time.clock_gettime_ns(self.clock_id)/1e9 if self.clock_id is not None
               else (int(fields[11])+int(fields[12]))/self.ticks)
        return dict(monotonic=time.monotonic(), epoch=time.time(), cpu_seconds=cpu,
                    rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'), pid=self.pid)

    def sample(self, interval):
        try:
            with (self.directory/'process_samples.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['monotonic','epoch','cpu_seconds','rss_bytes','pid'])
                writer.writeheader()
                while not self.done.is_set():
                    start = time.monotonic()
                    writer.writerow(self.point())
                    self.done.wait(max(0., interval-(time.monotonic()-start)))
        except Exception as exc:
            self.failure = f'{type(exc).__name__}: {exc}'

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError('CPU sampler did not stop')
            self.thread = None
        if self.collector:
            if self.collector.poll() is None:
                self.collector.terminate()
                try:
                    self.collector.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.collector.kill()
                    self.collector.wait(timeout=5)
            self.collector = None
        if self.log:
            self.log.close()
            self.log = None

    def handle(self, message):
        action = message['action']
        if action == 'init':
            candidates = ([int(message['pid'])] if message.get('pid') is not None else
                          [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()
                           and (p/'comm').exists() and (p/'comm').read_text().strip() == 'virtuoso-t'])
            if len(candidates) != 1:
                raise ValueError(f'Expected one database PID; found {candidates}; specify --pid')
            self.pid = candidates[0]
            if Path(f'/proc/{self.pid}/comm').read_text().strip() != 'virtuoso-t':
                raise ValueError('Selected PID is not virtuoso-t')
            self.ticks = os.sysconf('SC_CLK_TCK')
            fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()
            self.identity = int(fields[19])
            self.clock_id = None
            candidate_clock = ctypes.c_int()
            libc = ctypes.CDLL(None)
            if libc.clock_getcpuclockid(self.pid, ctypes.byref(candidate_clock)) == 0:
                try:
                    time.clock_gettime_ns(candidate_clock.value)
                    self.clock_id = candidate_clock.value
                except OSError:
                    pass
            if not shutil.which('collectl'):
                raise RuntimeError('Install collectl on the database host before profiling')
            return dict(hostname=os.uname().nodename, pid=self.pid, start_ticks=self.identity,
                        cpu_counter='process_cpu_clock' if self.clock_id is not None else 'proc_stat_ticks',
                        counter_resolution_s=time.clock_getres(self.clock_id) if self.clock_id is not None else 1/self.ticks,
                        collectl=shutil.which('collectl'), sample=self.point())
        if action == 'sample':
            if self.failure:
                raise RuntimeError(self.failure)
            if self.collector and self.collector.poll() is not None:
                raise RuntimeError('collectl exited unexpectedly; inspect collector.log')
            return self.point()
        if action == 'start':
            if self.directory is not None:
                raise RuntimeError('Release the previous recording before starting another')
            self.directory = Path(tempfile.mkdtemp(prefix='wrk-gen-'))
            config = self.directory/'collectl.conf'
            config.write_text('')
            self.log = (self.directory/'collector.log').open('w')
            interval = float(message['collectl_interval'])
            command = ['collectl','-C',str(config),'-sZ','-i',f'{interval}:{interval}',
                       '--procfilt',f'p{self.pid}','-f',str(self.directory/'collect'),'-F0']
            self.collector = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                             env=dict(os.environ,TZ='UTC',LC_ALL='C'))
            self.done.clear()
            self.failure = None
            self.thread = threading.Thread(target=self.sample, args=(float(message['cpu_interval']),), daemon=True)
            self.thread.start()
            return dict(directory=str(self.directory), command=command)
        if action == 'stop':
            self.stop()
            if self.failure:
                raise RuntimeError(self.failure)
            files = []
            for path in sorted(self.directory.iterdir()):
                if path.is_file():
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    files.append(dict(name=path.name, size=path.stat().st_size, sha256=digest))
            if not any('.raw' in item['name'] for item in files):
                raise RuntimeError(f'collectl produced no raw recording; retained {self.directory}')
            return files
        if action == 'fetch':
            name = message['name']
            if Path(name).name != name:
                raise ValueError('Invalid recording name')
            with (self.directory/name).open('rb') as handle:
                handle.seek(int(message['offset']))
                return base64.b64encode(handle.read(65536)).decode()
        if action == 'release':
            self.stop()
            shutil.rmtree(self.directory)
            self.directory = None
            return True
        if action == 'close':
            self.stop()
            return True
        raise ValueError('Unknown collector action')


def main():
    agent = Agent()
    try:
        for line in sys.stdin:
            message = {}
            try:
                message = json.loads(line)
                result = dict(ok=True, result=agent.handle(message))
            except Exception as exc:
                result = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
            print(json.dumps(result,allow_nan=False),flush=True)
            if message.get('action') == 'close':
                break
    finally:
        agent.stop()


if __name__ == '__main__':
    main()
