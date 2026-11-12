"""Read-only screening of QC observations censored by recorded request deadlines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .queries import fingerprint


TIMEOUT_STATUSES = {'timeout', 'partial_results', 'execution_timeout', 'query_timeout'}
AUDIT_COLUMNS = ('query_id', 'query_text', 'runtime_seconds', 'exclusion_reason',
                 'execution_statuses', 'server_timeout_seconds', 'client_timeout_seconds',
                 'effective_timeout_seconds', 'run', 'manifest_path')


def timeout_audit(qc_path):
    """Treat elapsed times at/above a collection deadline as censored, even if status is ok.

    This is conservative: HTTP 200 without timeout headers cannot establish that
    a result returned near its server deadline completed. Unknown deadline
    provenance for an otherwise usable measurement fails explicitly.
    """
    raw = pd.read_csv(qc_path, dtype=str, keep_default_na=False)
    key = next((key for key in ('query_id', 'query_no', 'id') if key in raw), None)
    text = next((key for key in ('query_text', 'text', 'queryString') if key in raw), None)
    if key is None or text is None or 'runtime_seconds' not in raw:
        raise ValueError('Timeout screening requires QC query IDs, texts and runtime_seconds')
    manifests, hashes, excluded = {}, {}, []
    for row in raw.to_dict('records'):
        statuses = set(json.loads(row.get('execution_statuses') or '[]'))
        statuses.add(row.get('status', ''))
        value = pd.to_numeric(row['runtime_seconds'], errors='coerce')
        explicit = bool(statuses & TIMEOUT_STATUSES)
        if pd.isna(value) and not explicit:
            continue
        run = row.get('run', '')
        if not run and row.get('status') == 'success' and (Path(qc_path).parent / 'manifest.json').is_file():
            run = str(Path(qc_path).parent)
        if explicit and not run:
            excluded.append(dict(query_id=row[key], query_text=row[text], runtime_seconds=value,
                exclusion_reason='recorded_timeout_or_partial_results'))
            continue
        if not run:
            if row.get('status', '') in ('measured', 'ok', 'success', '') and pd.notna(value):
                raise ValueError(f'Missing QC timeout provenance: {row[key]}')
            continue
        path = Path(run) / 'manifest.json'
        if run not in manifests:
            if not path.is_file():
                raise ValueError(f'Missing QC collection manifest for timeout screening: {path}')
            hashes[str(path)] = fingerprint(path)
            manifest = json.loads(path.read_text())
            server = float(manifest.get('server_query_timeout_seconds') or manifest.get('query_timeout_seconds') or 0)
            client = float(manifest.get('timeout_seconds') or 0)
            if not np.isfinite([server, client]).all() or min(server, client) < 0:
                raise ValueError(f'Invalid recorded request deadline: {path}')
            positive = [limit for limit in (server, client) if limit > 0]
            manifests[run] = server, client, min(positive) if positive else None
        server, client, limit = manifests[run]
        at_limit = limit is not None and pd.notna(value) and value >= limit
        if explicit or at_limit:
            excluded.append(dict(query_id=row[key], query_text=row[text], runtime_seconds=value,
                exclusion_reason='recorded_timeout_or_partial_results' if explicit else 'elapsed_at_or_above_recorded_timeout',
                execution_statuses=json.dumps(sorted(statuses-{''})), server_timeout_seconds=server,
                client_timeout_seconds=client, effective_timeout_seconds=limit, run=run, manifest_path=str(path)))
    if any(fingerprint(path) != digest for path, digest in hashes.items()):
        raise ValueError('QC timeout provenance changed while screening')
    return pd.DataFrame(excluded, columns=AUDIT_COLUMNS), hashes
