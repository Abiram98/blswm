"""All output paths are constrained to the startup working directory."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd().resolve()

def output(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Output escapes startup directory {ROOT}: {path}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def read(path):
    p = Path(path)
    if p.suffix == '.csv':
        with p.open(newline='') as f:
            return list(csv.DictReader(f))
    if p.suffix == '.jsonl':
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())

def write(path, data):
    p = output(path)
    if p.suffix == '.csv':
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]) if data else [])
            w.writeheader()
            w.writerows(data)
    elif p.suffix == '.jsonl':
        p.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in data))
    else:
        p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
