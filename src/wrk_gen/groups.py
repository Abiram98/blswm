"""Offline grouping for isolated query-log profiling."""
import numpy as np
from pathlib import Path
from .queries import load_queries, new_directory, fingerprint, write_json, write_queries


def create_groups(input_path, output, group_size, sample_size=None, seed=42,
                  runtime_min=None, runtime_max=None, runtime_column=None):
    if group_size < 1 or sample_size is not None and sample_size < 1:
        raise ValueError('group-size and sample-size must be positive')
    queries = load_queries(input_path, runtime_column)
    eligible = [q for q in queries if (runtime_min is None or q['runtime_seconds'] is not None and q['runtime_seconds'] >= runtime_min)
                and (runtime_max is None or q['runtime_seconds'] is not None and q['runtime_seconds'] <= runtime_max)]
    if sample_size is not None:
        if sample_size > len(eligible):
            raise ValueError(f'Need {sample_size} queries; only {len(eligible)} are eligible')
        indices = np.random.default_rng(seed).choice(len(eligible), sample_size, replace=False)
        eligible = [eligible[int(i)] for i in indices]
    if not eligible:
        raise ValueError('No eligible queries')
    source = Path(input_path).resolve()
    source = source/'ql.csv' if source.is_dir() else source
    source_hash = fingerprint(source)
    destination = new_directory(output, [source.parent] if source.name == 'ql.csv' else [])
    groups = []
    for index, start in enumerate(range(0, len(eligible), group_size), 1):
        folder = destination/f'group_{index:04}'
        folder.mkdir()
        selected = eligible[start:start+group_size]
        write_queries(folder/'ql.csv', selected)
        groups.append(dict(path=folder.name, queries=len(selected), query_ids=[q['query_id'] for q in selected],
                           sha256=fingerprint(folder/'ql.csv')))
    manifest = dict(schema_version=1, source=str(source), source_sha256=source_hash, source_queries=len(queries),
                    selected_queries=len(eligible), group_size=group_size, seed=seed, sample_size=sample_size,
                    runtime_min=runtime_min, runtime_max=runtime_max, runtime_column=runtime_column,
                    groups=groups)
    if fingerprint(source) != source_hash:
        raise ValueError('Query log changed during grouping')
    write_json(destination/'groups.json', manifest)
    return manifest


def group_paths(paths):
    result = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if (path/'groups.json').is_file():
            import json
            manifest = json.loads((path/'groups.json').read_text())
            for group in manifest['groups']:
                folder = (path/group['path']).resolve()
                if not folder.is_relative_to(path) or fingerprint(folder/'ql.csv') != group['sha256']:
                    raise ValueError('Profiling group changed or escaped its manifest')
                result.append(folder)
        else:
            if path.is_file() and path.name != 'ql.csv':
                raise ValueError('Individual profiling inputs must be group directories or ql.csv files')
            result.append(path if path.is_dir() else path.parent)
    if not result or len(set(result)) != len(result):
        raise ValueError('Provide distinct query group paths')
    for path in result:
        load_queries(path)
    return result
