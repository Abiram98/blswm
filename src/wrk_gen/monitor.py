"""Owned local/SSH collector sessions and verified recording transfer."""
import base64
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time


class Monitor:
    def __init__(self, host=None, user='root', port=22, pid=None,
                 identity_file=None, password_env='WRK_GEN_SSH_PASSWORD'):
        self.files = tempfile.TemporaryDirectory(prefix='wrk-gen-ssh-')
        self.errors = (Path(self.files.name)/'ssh.log').open('w+')
        self.messages = queue.Queue()
        self.child = None
        self.recording = False
        source = files('wrk_gen').joinpath('remote_agent.py').read_text()
        env = dict(os.environ)
        if host is None:
            command = [sys.executable,'-u','-c',source]
        else:
            if not re.fullmatch(r'[A-Za-z0-9_.:\[\]-]+',host) or host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.-]+',user) or user.startswith('-'):
                raise ValueError('Invalid SSH host/user')
            command = ['ssh','-T','-p',str(port),'-o','ConnectTimeout=5','-o','ServerAliveInterval=10',
                       '-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=accept-new']
            if identity_file:
                command += ['-i',str(Path(identity_file).expanduser())]
            if password_env in env:
                helper = Path(self.files.name)/'askpass'
                helper.write_text('#!'+sys.executable+'\nimport os\nprint(os.environ["WRK_GEN_ASKPASS_SECRET"])\n')
                helper.chmod(0o700)
                env.update(SSH_ASKPASS=str(helper),SSH_ASKPASS_REQUIRE='force',DISPLAY='wrk-gen:0',
                           WRK_GEN_ASKPASS_SECRET=env.pop(password_env))
                command += ['-o','NumberOfPasswordPrompts=1','-o','PreferredAuthentications=password,keyboard-interactive']
            else:
                command += ['-o','BatchMode=yes']
            command += [f'{user}@{host}','python3 -u -c '+shlex.quote(source)]
        try:
            self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=self.errors, text=True, env=env, bufsize=1)
            def receive():
                try:
                    for line in self.child.stdout:
                        self.messages.put(line)
                finally:
                    self.messages.put(None)
            self.reader = threading.Thread(target=receive,daemon=True)
            self.reader.start()
            self.info = self.rpc('init',pid=pid)
            samples = []
            for _ in range(5):
                start = time.monotonic()
                point = self.rpc('sample')
                end = time.monotonic()
                samples.append(dict(offset=(start+end)/2-point['monotonic'], uncertainty_s=(end-start)/2))
            self.calibration = min(samples,key=lambda s:s['uncertainty_s'])
        except BaseException:
            self.close()
            raise

    def rpc(self, action, **kwargs):
        self.child.stdin.write(json.dumps(dict(action=action,**kwargs))+'\n')
        self.child.stdin.flush()
        try:
            line = self.messages.get(timeout=60)
        except queue.Empty as exc:
            raise RuntimeError(f'Collector RPC timed out: {action}') from exc
        if line is None:
            self.errors.flush()
            self.errors.seek(0)
            raise RuntimeError('Collector disconnected: '+self.errors.read()[-1500:])
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(result.get('error','Collector RPC failed'))
        return result['result']

    def start(self, cpu_interval, collectl_interval):
        info = self.rpc('start',cpu_interval=cpu_interval,collectl_interval=collectl_interval)
        self.recording = True
        return info

    def stop(self, destination):
        files = self.rpc('stop')
        self.recording = False
        destination = Path(destination)
        destination.mkdir()
        for item in files:
            name = item['name']
            if Path(name).name != name:
                raise ValueError('Unsafe recording name')
            path = destination/name
            digest = hashlib.sha256()
            offset = 0
            with path.open('wb') as handle:
                while offset<item['size']:
                    data = base64.b64decode(self.rpc('fetch',name=name,offset=offset),validate=True)
                    if not data:
                        raise RuntimeError('Truncated collector recording')
                    handle.write(data)
                    digest.update(data)
                    offset += len(data)
            if digest.hexdigest()!=item['sha256'] or offset!=item['size']:
                raise RuntimeError('Collector recording checksum mismatch')
        self.rpc('release')
        return files

    def close(self):
        if self.child:
            if self.child.poll() is None:
                try:
                    self.rpc('close')
                except Exception:
                    pass
                try:
                    self.child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.child.terminate()
                    try:
                        self.child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.child.kill()
                        self.child.wait()
            self.child = None
        self.errors.close()
        self.files.cleanup()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
