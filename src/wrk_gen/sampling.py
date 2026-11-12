"""Seeded composition sampling with user-supplied CSVs, profiles and counts."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .bag import QueryBag, QueryOccurrence, canonical, validate_composition
from .measured_catalog import MeasuredCatalog, json_values
from .profiles import parse_profiles
from .queries import fingerprint
from .timeout_screening import timeout_audit

@dataclass(frozen=True)
class CompositionResult:
    bag: QueryBag | None
    reason: str | None
    details: dict

    @property
    def available(self):
        return self.bag is not None


class CompositionSampler:
    @staticmethod
    def read_exclusions(paths):
        ids, texts, sources = set(), set(), {}
        for value in paths:
            path = Path(value).resolve()
            digest = fingerprint(path)
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
            key = next((key for key in ('query_id', 'id', 'query_no') if key in frame), None)
            if key is None or frame.empty or frame[key].str.strip().eq('').any():
                raise ValueError(f'Exclusion CSV needs nonempty query IDs: {path}')
            ids.update(frame[key])
            text = next((key for key in ('query_text', 'text', 'queryString') if key in frame), None)
            if text:
                texts.update(frame.loc[frame[text].str.strip().ne(''), text])
            if fingerprint(path) != digest:
                raise ValueError(f'Exclusion CSV changed while reading: {path}')
            sources[str(path)] = digest
        return ids, texts, sources

    @staticmethod
    def prepare_catalog(catalog, *, excluded_ids=(), excluded_texts=()):
        data = catalog.queries.copy(deep=True)
        if 'query_text' not in data or data.query_id.duplicated().any():
            raise ValueError('Catalog needs query text and unique query IDs')
        data = data[~data.query_id.isin(excluded_ids) & ~data.query_text.isin(excluded_texts)]
        data = data[data.query_text.notna() & data.query_text.str.strip().ne('')]
        return data.sort_values('query_id').drop_duplicates('query_text')

    @staticmethod
    def sample_queries(available, count, rng, *, role):
        if len(available) < count:
            raise ValueError(f'Need {count} {role} queries; have {len(available)}')
        return available.iloc[rng.choice(len(available), count, replace=False)].copy()

    @staticmethod
    def sample_balanced_occurrences(available, count, rng, *, role):
        """Use each distinct candidate before reuse, balancing counts to within one."""
        if count < 1 or available.empty:
            raise ValueError(f'Need a positive count and at least one {role} candidate')
        cycles = []
        remaining = count
        while remaining:
            take = min(remaining, len(available))
            cycles.extend(rng.permutation(len(available))[:take])
            remaining -= take
        return available.iloc[np.asarray(cycles)].copy().reset_index(drop=True)

    def __init__(self, profiles):
        self.profiles = {profile.id: profile for profile in profiles}
        if not self.profiles:
            raise ValueError('At least one execution profile is required')

    def sample(self, composition, qc_paths, *, excluded_ql_paths=(), exclude_timeout_censored=False):
        if type(exclude_timeout_censored) is not bool:
            raise ValueError('exclude_timeout_censored must be boolean')
        validate_composition(composition, self.profiles)
        paths = tuple(Path(path).expanduser().resolve() for path in qc_paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Supply one or more different QC CSV paths')
        ids, texts, hashes = self.read_exclusions(excluded_ql_paths)
        frames, windows, timeout_exclusions = [], set(), []
        active = [self.profiles[key] for key, count in composition['counts'].items() if count]
        for index, path in enumerate(paths):
            hashes[str(path)] = fingerprint(path)
            if exclude_timeout_censored:
                audit, provenance = timeout_audit(path)
                hashes.update(provenance)
                ids.update(audit.query_id)
                texts.update(audit.query_text)
                timeout_exclusions.extend(audit.to_dict('records'))
            sidecar = path.parent / 'qc_analysis.json'
            if sidecar.exists():
                hashes[str(sidecar)] = fingerprint(sidecar)
            catalog = MeasuredCatalog.from_csv(path)
            if catalog.cpu_peak_window_seconds is not None:
                windows.add(catalog.cpu_peak_window_seconds)
            data = self.prepare_catalog(catalog, excluded_ids=ids, excluded_texts=texts)
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
            source_id = next(key for key in ('query_id', 'query_no', 'id') if key in raw)
            if 'mean_cpu_pct' in raw:
                values = raw.set_index(source_id)['mean_cpu_pct'].replace('', np.nan)
                means = pd.to_numeric(data.query_id.map(values), errors='raise')
                if (means.notna() & (~np.isfinite(means) | means.lt(0))).any():
                    raise ValueError('mean_cpu_pct must be finite and nonnegative or missing')
                data['mean_cpu_pct'] = means.where(data.runtime_seconds.notna())
                if 'mean_cpu_status' in raw:
                    status = data.query_id.map(raw.set_index(source_id).mean_cpu_status)
                    data['mean_cpu_pct'] = data.mean_cpu_pct.where(status.isin(('', 'measured')))
            if any('mean_cpu_pct' in profile.to_dict() for profile in active) and 'mean_cpu_pct' not in data:
                raise ValueError('A mean-CPU profile requires measured mean_cpu_pct in the input CSV')
            data['source_query_id'] = data.query_id
            data['source_catalog'] = str(path)
            data['query_id'] = [f'catalog{index}:{value}' for value in data.query_id]
            frames.append(data)
        if len(windows) > 1:
            raise ValueError('QC sources have incompatible CPU peak windows')
        combined = pd.concat(frames, ignore_index=True)
        # A later QC source may identify a censored alias of an earlier source's text.
        combined = combined[~combined.source_query_id.isin(ids) & ~combined.query_text.isin(texts)]
        prepared = self.prepare_catalog(MeasuredCatalog(combined))
        pools = {key: prepared[profile.eligible(prepared)] for key, profile in self.profiles.items()}
        counts = {key: len(pool) for key, pool in pools.items()}
        rng = np.random.default_rng(composition['sampling_seed'])
        # Maximum distinct assignment across overlapping profiles. An augmenting
        # path moves earlier assignments when a narrower profile needs their text.
        candidates = {key: pool.iloc[rng.permutation(len(pool))] for key, pool in pools.items()
                      if composition['counts'].get(key, 0)}
        owners = {}
        def augment(key, visited):
            if key in visited:
                return False
            visited.add(key)
            for query_id in candidates[key].query_id:
                previous = owners.get(query_id)
                if previous is None or previous != key and augment(previous, visited):
                    owners[query_id] = key
                    return True
            return False
        for key, count in composition['counts'].items():
            for _ in range(count):
                if not augment(key, set()):
                    break
        selected, used = [], set()
        for key, count in composition['counts'].items():
            if not count:
                continue
            pool = candidates[key]
            draws = pool[pool.query_id.map(owners).eq(key)]
            if pool.empty or composition.get('reuse', 'prefer_distinct') == 'distinct_only' and len(draws) < count:
                return CompositionResult(None, 'insufficient_distinct_queries' if not pool.empty else 'no_eligible_queries',
                                         dict(profile=key, required=count, available=len(pool), eligible_counts=counts))
            if len(draws) < count:
                extra = self.sample_balanced_occurrences(pool, count - len(draws), rng, role=key)
                draws = pd.concat([draws, extra], ignore_index=True)
            used.update(draws.query_text)
            draws = draws.copy()
            draws['query_type'] = key
            selected.append(draws)
        occurrences = []
        for index, row in enumerate(pd.concat(selected, ignore_index=True).to_dict('records'), 1):
            metrics = {key: value for key, value in row.items() if key not in
                       ('position', 'runtime_band', 'cpu_band', 'query_id', 'query_text', 'query_type',
                        'source_query_id', 'source_catalog')}
            occurrences.append(QueryOccurrence(f'{composition["id"]}__occurrence_{index:06d}',
                row['source_catalog'], row['source_query_id'], row['query_text'], row['query_type'],
                canonical(json_values(metrics))))
        if any(fingerprint(path) != value for path, value in hashes.items()):
            raise ValueError('Source changed during composition')
        provenance = dict(input_sha256=hashes, eligible_counts=counts,
                          cpu_peak_window_seconds=next(iter(windows), None),
                          sampling_protocol='Seeded maximum distinct assignment across profiles, then balanced cycles')
        if exclude_timeout_censored:
            provenance['timeout_screening'] = dict(enabled=True, excluded_records=len(timeout_exclusions),
                excluded_query_ids=sorted({row['query_id'] for row in timeout_exclusions}),
                excluded_texts=len({row['query_text'] for row in timeout_exclusions}),
                rule='Exclude explicit timeout/partial observations and elapsed times at/above recorded collection deadlines, by ID and exact text')
        bag = QueryBag.create(occurrences, {'profiles': [profile.to_dict() for profile in self.profiles.values()]},
            composition, provenance)
        bag.validate()
        return CompositionResult(bag, None, dict(eligible_counts=counts, occurrences=len(occurrences),
                                 distinct_query_texts=len(used)))

def sample_workload(input_path, output, *, count=None, recipe=None, profiles=None, composition=None,
                    group_counts=(), seed=None, exclusions=(), allow_reuse=False, reuse=None,
                    name=None, dataset=None, screen_timeouts=False,
                    runtime_min=None, runtime_max=None, cpu_min=None, cpu_max=None):
    """All experimental settings come from arguments or explicit JSON inputs."""
    if recipe and (profiles or composition or count is not None):
        raise ValueError('Use --recipe or --profiles/--composition/--count')
    configuration_hashes = {}
    def read(path):
        path = Path(path).expanduser().resolve()
        configuration_hashes[str(path)] = fingerprint(path)
        return json.loads(path.read_text())
    if recipe:
        config = read(recipe)
        if set(config) != {'profiles', 'groups'}:
            raise ValueError('A recipe requires profiles and ordered groups')
        definitions = parse_profiles({'profiles': config['profiles']})
        counts = {}
        for group in config['groups']:
            if set(group) != {'profile', 'count'} or group['profile'] in counts:
                raise ValueError('Recipe groups require unique profile names and counts')
            counts[group['profile']] = group['count']
        settings = dict(id='workload', dataset='custom', counts=counts, sampling_seed=42,
                        reuse='distinct_only')
    else:
        if profiles:
            definitions = parse_profiles(read(profiles))
        else:
            definition = dict(id='selected')
            for metric, lower, upper in [('runtime_seconds', runtime_min, runtime_max),
                                         ('peak_cpu_pct', cpu_min, cpu_max)]:
                if lower is not None or upper is not None:
                    definition[metric] = dict(min=lower, max=upper)
            definitions = parse_profiles({'profiles': [definition]})
        if composition:
            if count is not None:
                raise ValueError('Use --count or --composition')
            settings = read(composition)
        else:
            if count is not None and (profiles or group_counts):
                raise ValueError('--count selects one default group; use --group with --profiles')
            settings = dict(id='workload', dataset='custom',
                            counts={'selected': count} if count is not None else {},
                            sampling_seed=42, reuse='distinct_only')
    for item in group_counts:
        key, separator, value = item.partition('=')
        if not separator or not value.isdigit():
            raise ValueError('--group must be PROFILE=COUNT with a nonnegative integer count')
        settings['counts'][key] = int(value)
    for key, value in [('sampling_seed', seed), ('id', name), ('dataset', dataset), ('reuse', reuse)]:
        if value is not None:
            settings[key] = value
    if allow_reuse:
        if reuse == 'distinct_only':
            raise ValueError('--allow-reuse conflicts with --reuse distinct_only')
        settings['reuse'] = 'prefer_distinct'
    paths = [input_path] if isinstance(input_path, (str, Path)) else input_path
    paths = [Path(path).expanduser().resolve() for path in paths]
    paths = [path / 'catalog.csv' if path.is_dir() else path for path in paths]
    result = CompositionSampler(definitions).sample(settings, paths, excluded_ql_paths=exclusions,
                                                    exclude_timeout_censored=screen_timeouts)
    if not result.available:
        raise ValueError(f'{result.reason}: {result.details}')
    if any(fingerprint(path) != digest for path, digest in configuration_hashes.items()):
        raise ValueError('Sampling configuration changed while reading')
    destination = result.bag.save(output)
    return dict(output=str(destination), bag_id=result.bag.id, **result.details,
                profile_counts=settings['counts'])
