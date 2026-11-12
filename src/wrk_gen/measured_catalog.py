"""Normalize scalar or repeated measurements without altering query text."""
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .queries import fingerprint

METRICS = {
    'latency': ('Execution time', 's', ('runtime_seconds', 'elapsed_seconds', 'mean_elapsed_seconds',
                                     'latency', 'execution_time', 'ex time (secs)')),
    'cpu': ('CPU', '%', ('peak_cpu_pct', 'cpu')),
    'memory': ('Memory', 'MiB', ('peak_memory_mib', 'memory')),
}
STATISTICS = ['count', 'missing', 'mean', 'std', 'min', 'p25', 'median', 'p75', 'p95', 'max']
MISSING = {'', 'nan', 'na', 'n/a', 'null', 'none'}


def metric_columns(columns, candidates):
    """Resolve an exact column or a family such as runtime_seconds_rep_N."""
    for candidate in candidates:
        if candidate in columns:
            return [candidate]
        pattern = re.compile(re.escape(candidate) + r'_rep_(\d+)$')
        matches = [(int(match[1]), column) for column in columns
                   if (match := pattern.fullmatch(column))]
        if matches:
            return [column for _, column in sorted(matches)]
    raise ValueError(f'Missing resource column; expected one of: {", ".join(candidates)} '
                     '(or columns with _rep_N suffixes)')


def included_rows(frame, suffix):
    """Honor the peak mapper's status and boundary-counter exclusions."""
    included = pd.Series(True, index=frame.index)
    status = 'status' + suffix
    if status in frame:
        included &= frame[status].str.strip().eq('ok')
    inclusion = 'included_in_summary' + suffix
    if inclusion in frame:
        included &= frame[inclusion].str.strip().str.lower().isin(['true', '1'])
    flags = 'flags' + suffix
    if flags in frame:
        included &= ~frame[flags].str.contains('missing_boundary_counters', regex=False)
    return included


def runtime_band(values):
    return pd.cut(values, [0, .1, 1, 10, np.inf], right=False,
                  labels=['<0.1 s', '[0.1,1) s', '[1,10) s', '>=10 s']).astype('string')


def cpu_band(values):
    result = pd.Series('unresolved', index=values.index, dtype='string')
    for label, mask in [('>1500%', values > 1500), ('(200,1500]%', (values > 200) & (values <= 1500)),
                        ('[100,200]%', (values >= 100) & (values <= 200)), ('<100%', values < 100)]:
        result.loc[mask] = label
    return result


def json_values(value):
    """Keep unavailable statistics as JSON null, never NaN."""
    return json.loads(json.dumps(value, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)),
                      parse_constant=lambda _: None)


class MeasuredCatalog:
    """One row per query occurrence, in input order, with independent resource means.

    Wide inputs average available eligible repetitions per metric. Scalar inputs
    retain their recorded values. Missing CPU peaks describe measurement coverage,
    not zero CPU demand. Sampling preserves both resource bands and their positions.
    """

    def __init__(self, queries, *, name='workload', cpu_peak_window_seconds=None, metadata=None):
        if cpu_peak_window_seconds is not None:
            if not math.isfinite(cpu_peak_window_seconds) or cpu_peak_window_seconds <= 0:
                raise ValueError("CPU peak window must be positive and finite")
        self.name = name
        self.cpu_peak_window_seconds = cpu_peak_window_seconds
        self.metadata = dict(metadata or {})
        self.queries = queries.copy().reset_index(drop=True)
        if self.queries.empty:
            raise ValueError('The workload has no queries')
        for field in ('query_id', 'runtime_seconds', 'peak_cpu_pct'):
            if field not in self.queries:
                raise ValueError(f'Missing query field: {field}')
        if self.queries.query_id.isna().any() or self.queries.query_id.astype(str).str.strip().eq('').any():
            raise ValueError('Query IDs must be present')
        self.queries['query_id'] = self.queries.query_id.astype(str)
        for field in ('runtime_seconds', 'peak_cpu_pct'):
            values = pd.to_numeric(self.queries[field], errors='raise').astype(float)
            if (values.notna() & (~np.isfinite(values) | values.lt(0))).any():
                raise ValueError(f'{field} must contain finite, nonnegative measurements or missing values')
            self.queries[field] = values
        self.queries['position'] = np.arange(1, len(self.queries) + 1)
        self.queries['runtime_band'] = runtime_band(self.queries.runtime_seconds)
        self.queries['cpu_band'] = cpu_band(self.queries.peak_cpu_pct)

    @classmethod
    def from_csv(cls, path, *, name='workload', cpu_peak_window_seconds=None):
        path = Path(path).resolve()
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        query_id = next((key for key in ('query_id', 'query_no', 'id') if key in frame), None)
        if query_id is None:
            raise ValueError('Measurements require a query_id, query_no, or id column')
        normalized = frame.copy()
        for key in normalized:
            if re.fullmatch(r'status(?:_rep_\d+)?', key):
                normalized[key] = normalized[key].replace({'measured': 'ok', 'success': 'ok'})
        queries = pd.DataFrame({'query_id': frame[query_id]})
        text = next((key for key in ('query_text', 'text', 'queryString') if key in frame), None)
        if text:
            queries['query_text'] = frame[text]
        selected = {}
        for metric, output in [('latency', 'runtime_seconds'), ('cpu', 'peak_cpu_pct'), ('memory', 'peak_memory_mib')]:
            if metric == 'memory' and not any(key.startswith('peak_memory_mib') for key in frame):
                continue
            columns = metric_columns(frame.columns, METRICS[metric][2])
            selected[metric] = columns
            measurements = pd.DataFrame(index=frame.index)
            for column in columns:
                match = re.search(r'_rep_\d+$', column)
                suffix = match[0] if match else ''
                values = frame[column].str.strip()
                values = values.mask(values.str.lower().isin(MISSING)).where(included_rows(normalized, suffix))
                peak_status = 'cpu_peak_status' + suffix
                if metric == 'cpu' and peak_status in frame:
                    values = values.where(frame[peak_status].isin(['', 'measured']))
                numbers = pd.to_numeric(values, errors='raise')
                if (numbers.notna() & (~np.isfinite(numbers) | numbers.lt(0))).any():
                    raise ValueError(f'Invalid measurement in {column}')
                measurements[column] = numbers
            queries[output] = measurements.mean(axis=1)
        for key in ('row_number', 'status', 'cpu_peak_status', 'run', 'workload', 'workload_row', 'flags', 'peak_rss_bytes'):
            if key in frame:
                queries[key] = frame[key]
        settings = [key for key in frame if re.fullmatch(r'(?:cpu_peak_window_seconds|cpu_window_seconds)(?:_rep_\d+)?', key)]
        windows = {float(value) for key in settings for value in frame[key]
                   if value.strip().lower() not in MISSING}
        if not windows:
            sidecar = path.parent / 'qc_analysis.json'
            if sidecar.exists():
                receipt = json.loads(sidecar.read_text())
                datasets = receipt.get('datasets', {})
                if path.name in {f'{dataset}_query_characteristics.csv' for dataset in datasets}:
                    windows = {float(receipt['settings']['cpu_peak_window'])}
        if len(windows) > 1:
            raise ValueError('Measurements contain different CPU peak windows')
        if windows:
            recorded = next(iter(windows))
            if cpu_peak_window_seconds is not None and not math.isclose(recorded, cpu_peak_window_seconds):
                raise ValueError('Requested CPU peak window differs from recorded measurements; recalculate peaks first')
            cpu_peak_window_seconds = recorded
        return cls(queries, name=name, cpu_peak_window_seconds=cpu_peak_window_seconds,
                   metadata=dict(source=str(path), source_sha256=fingerprint(path), selected_columns=selected,
                                 aggregation='Mean of available eligible repetitions per metric; scalar values unchanged'))

