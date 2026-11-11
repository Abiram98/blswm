import math
import hashlib
from .io import read

def identity(text):
    # Conservative: exact stripped text avoids changing literal whitespace semantics.
    return hashlib.sha256(text.strip().encode()).hexdigest()

def profile(runtime, cpu):
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

def load(path):
    rows = read(path)
    seen = set()
    for r in rows:
        if r['query_id'] in seen:
            raise ValueError('Duplicate query_id')
        seen.add(r['query_id'])
        if not r.get('query_text'):
            from pathlib import Path
            r['query_text'] = (Path(path).parent / r['query_file']).read_text()
        t = float(r['isolated_runtime_s'])
        if not math.isfinite(t) or t < 0:
            raise ValueError('Invalid runtime')
        r['isolated_runtime_s'] = t
        r['profile'] = profile(t, r.get('peak_cpu_pct')) if r.get('status', 'success') == 'success' else None
    return rows
