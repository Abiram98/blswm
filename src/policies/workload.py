"""Read ql.csv plus its paired, one-based arrival_time.csv without profiling labels."""
import csv
import math
from pathlib import Path


def load_workload(directory, arrivals=None):
    directory = Path(directory)
    with (directory / 'ql.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'id', 'text'}.issubset(reader.fieldnames or []):
            raise ValueError('ql.csv requires id and text columns')
        queries = list(reader)
    with Path(arrivals or directory / 'arrival_time.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'query_no', 'arrival_time'}.issubset(reader.fieldnames or []):
            raise ValueError('Arrival CSV requires query_no and arrival_time columns')
        arrival_rows = list(reader)
    if not queries or len(queries) != len(arrival_rows):
        raise ValueError('Workload and arrival CSVs must be nonempty and have equal row counts')
    times = {}
    for row in arrival_rows:
        number = int(row['query_no'])
        value = float(row['arrival_time'])
        if number in times or number < 1 or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid or duplicate arrival row')
        times[number] = value
    if set(times) != set(range(1, len(queries)+1)):
        raise ValueError('query_no must cover each workload row exactly once (one-based)')
    if any(times[i] > times[i+1] for i in range(1, len(queries))):
        raise ValueError('Arrival times must be nondecreasing in workload order')
    catalog, trace, texts = [], [], {}
    for i, row in enumerate(queries):
        query_id, text = row.get('id'), row.get('text')
        if not query_id or not text or not text.strip():
            raise ValueError(f'Workload row {i+1} requires id and text')
        if query_id in texts and texts[query_id] != text:
            raise ValueError(f'Conflicting query text for id {query_id}')
        if query_id not in texts:
            catalog.append(dict(query_id=query_id, query_text=text))
            texts[query_id] = text
        trace.append(dict(occurrence_id=f'occurrence_{i+1:06}', query_id=query_id,
                          arrival_s=times[i+1], trace_index=i))
    return trace, catalog
