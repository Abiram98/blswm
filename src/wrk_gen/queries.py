"""Query-log normalization and policy-compatible CSV exports."""
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDS = ['id', 'text', 'source_query_id', 'runtime_seconds', 'peak_cpu_pct', 'mean_cpu_pct', 'query_type']


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def text_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()


def load_queries(path, runtime_column=None):
    path = Path(path).expanduser().resolve()
    path = path/'ql.csv' if path.is_dir() else path
    if path.suffix == '.json':
        payload = json.loads(path.read_text())
        rows = payload.get('queries', payload.get('occurrences')) if isinstance(payload, dict) else payload
    else:
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Query log is empty: {path}')
    result, identities = [], {}
    for index, raw in enumerate(rows, 1):
        key = next((raw[k] for k in ('query_id', 'id') if raw.get(k)), None)
        text = next((raw[k] for k in ('query_text', 'text', 'queryString') if raw.get(k)), None)
        if key is None or not isinstance(text, str) or not text.strip():
            raise ValueError(f'{path}: row {index} requires an ID and query text')
        key = str(key)
        if key in identities:
            if identities[key] != text:
                raise ValueError(f'Conflicting texts for query ID {key}')
            raise ValueError(f'Duplicate query ID {key}; give repeated occurrences distinct IDs')
        identities[key] = text
        runtime_keys = (runtime_column,) if runtime_column else ('runtime_seconds', 'isolated_runtime_s', 'ex time (secs)', 'mean_latency')
        runtime = next((finite(raw.get(k)) for k in runtime_keys if raw.get(k) not in (None, '')), None)
        result.append(dict(query_id=key, query_text=text, source_query_id=str(raw.get('source_query_id') or key),
                           runtime_seconds=runtime, peak_cpu_pct=finite(raw.get('peak_cpu_pct')),
                           mean_cpu_pct=finite(raw.get('mean_cpu_pct')), query_type=raw.get('query_type') or raw.get('profile') or 'unknown',
                           status=raw.get('status') or 'success', query_sha256=text_hash(text)))
    return result


def new_directory(path, inputs=()):
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    for value in inputs:
        source = Path(value).expanduser().resolve()
        if source.is_dir() and destination.is_relative_to(source):
            raise ValueError(f'Output must be outside the input directory: {source}')
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_queries(path, queries):
    write_csv(path, [dict(id=q['query_id'], text=q['query_text'], source_query_id=q['source_query_id'],
                         runtime_seconds=q.get('runtime_seconds'), peak_cpu_pct=q.get('peak_cpu_pct'),
                         mean_cpu_pct=q.get('mean_cpu_pct'), query_type=q.get('query_type', 'unknown')) for q in queries], FIELDS)
