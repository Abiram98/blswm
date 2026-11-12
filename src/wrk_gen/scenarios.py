"""P1-P4 occurrence planning independent of dataset size and source paths."""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
import json
from types import SimpleNamespace
import numpy as np
from .arrival_time import BurstArrivalTime, GroupArrivalGenerator, PoissonArrivalTime, arrival_from_dict
from .profiles import integer, object_fields
from .bag import QueryBag, canonical
from .measured_catalog import MeasuredCatalog

@dataclass(frozen=True)
class QueryGroup:
    id: str
    query_count: int
    arrival: object

@dataclass(frozen=True)
class Skip:
    reason: str
    details: dict


@dataclass(frozen=True)
class ScenarioPhase:
    id: str
    occurrence_ids: tuple[str, ...]
    arrival: PoissonArrivalTime | BurstArrivalTime

    def group(self):
        return QueryGroup(self.id, len(self.occurrence_ids), self.arrival)


@dataclass(frozen=True)
class ScenarioPlan:
    phases: tuple[ScenarioPhase, ...]
    config_json: str


@dataclass
class ScenarioResult:
    status: str
    workload: object | None = None
    plan: ScenarioPlan | None = None
    reason: str | None = None
    details: dict | None = None

    @classmethod
    def skipped(cls, skip):
        return cls('SKIPPED_INCOMPATIBLE', reason=skip.reason, details=skip.details)


def spread_occurrences(occurrences, counts, rng):
    """Prefer distinct texts per phase without replacing any frozen occurrence.

    Process the most frequent texts first, assigning to the largest remaining
    capacities. This also constructs distinct partitions whenever possible for
    identical candidate membership across phases.
    """
    bins, capacity = [[] for _ in counts], list(counts)
    by_text = {}
    for item in sorted(occurrences, key=lambda item: item.occurrence_id):
        by_text.setdefault(item.query_text, []).append(item)
    ties = {text: float(rng.random()) for text in sorted(by_text)}
    texts = sorted(by_text, key=lambda text: (-len(by_text[text]), ties[text], text))
    for text in texts:
        used = set()
        for item in by_text[text]:
            candidates = [index for index, count in enumerate(capacity) if count]
            fresh = [index for index in candidates if index not in used]
            candidates = fresh or candidates
            target = max(candidates, key=lambda index: (capacity[index], -index))
            bins[target].append(item)
            capacity[target] -= 1
            used.add(target)
    return [tuple(item.occurrence_id for item in (values[index] for index in rng.permutation(len(values))))
            for values in bins]


def profile_name(config, field, *, multiple=False):
    value = config.get(field)
    values = [value] if isinstance(value, str) else value if multiple and isinstance(value, list) else []
    if (not values or any(not isinstance(item, str) or not item.strip() for item in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'{field} must be an execution profile ID' + (' or a nonempty array of unique IDs' if multiple else ''))
    return tuple(values)


class ArrivalScenario(ABC):
    scenario_type = ''
    fields = ()

    def __init__(self, config):
        object_fields(config, ('id', 'type', 'order_seed', 'arrival_seed', 'first_arrival_rule',
                              'transition', 'distinct_within_phase', 'residual_phases', *self.fields), 'scenario')
        if config.get('type') != self.scenario_type:
            raise ValueError(f'Expected scenario type {self.scenario_type}')
        for key in ('order_seed', 'arrival_seed'):
            integer(config.get(key), key)
        if config.get('first_arrival_rule', 'initial_gap') not in ('initial_gap', 'zero'):
            raise ValueError('first_arrival_rule must be initial_gap or zero')
        if config.get('transition', 'previous_group_policy') != 'previous_group_policy':
            raise ValueError('Only previous_group_policy transitions are supported')
        if type(config.get('distinct_within_phase', False)) is not bool:
            raise ValueError('distinct_within_phase must be boolean')
        residual = config.get('residual_phases', [])
        if not isinstance(residual, list):
            raise ValueError('residual_phases must be an array')
        for phase in residual:
            object_fields(phase, ('profiles', 'arrival'), 'residual phase')
            if (not isinstance(phase.get('profiles'), list) or not phase['profiles']
                    or any(not isinstance(key, str) or not key for key in phase['profiles'])):
                raise ValueError('Residual phases need explicit profile IDs')
            arrival_from_dict(phase.get('arrival'))
        self.config = json.loads(canonical(config))
        self.validate_config()

    def validate_config(self):
        pass

    @abstractmethod
    def plan(self, bag: QueryBag, rng) -> ScenarioPlan | Skip:
        """Assign existing occurrences; selection is never available to a scenario."""

    def finish_plan(self, bag, phases, rng):
        phases = list(phases)
        used = {key for phase in phases for key in phase.occurrence_ids}
        for index, config in enumerate(self.config.get('residual_phases', []), 1):
            known = {profile['id'] for profile in json.loads(bag.profiles_json)['profiles']}
            if set(config['profiles']) - known:
                raise ValueError('Residual phase references an unknown execution profile')
            entries = [item for item in bag.occurrences if item.occurrence_id not in used
                       and item.profile_id in config['profiles']]
            if entries:
                ids = tuple(entries[index].occurrence_id for index in rng.permutation(len(entries)))
                phases.append(ScenarioPhase(f'residual_{index}', ids, arrival_from_dict(config['arrival'])))
                used.update(ids)
        missing = [item.occurrence_id for item in bag.occurrences if item.occurrence_id not in used]
        if missing:
            return Skip('unassigned_occurrences', dict(count=len(missing), occurrence_ids=missing))
        return ScenarioPlan(tuple(phases), canonical(self.config))

    def generate(self, bag, *, sources=None):
        bag.validate()
        plan = self.plan(bag, np.random.default_rng(self.config['order_seed']))
        if isinstance(plan, Skip):
            return ScenarioResult.skipped(plan)
        return self.materialize(bag, plan)

    def materialize(self, bag, plan):
        planned = [key for phase in plan.phases for key in phase.occurrence_ids]
        if Counter(planned) != Counter(item.occurrence_id for item in bag.occurrences):
            raise ValueError('Scenario plan must cover every frozen occurrence exactly once')
        if not plan.phases or any(not phase.occurrence_ids for phase in plan.phases):
            raise ValueError('Scenario phases must be nonempty')
        data = bag.frame().set_index('query_id', drop=False)
        if self.config.get('distinct_within_phase', False):
            for phase in plan.phases:
                if data.loc[list(phase.occurrence_ids)].query_text.duplicated().any():
                    return ScenarioResult.skipped(Skip('group_distinctness_infeasible', dict(phase=phase.id)))
        schedule = GroupArrivalGenerator(self.config['arrival_seed']).generate(
            [phase.group() for phase in plan.phases],
            first_arrival_rule=self.config.get('first_arrival_rule', 'initial_gap'))
        rows = data.loc[planned].reset_index(drop=True)
        rows['group_index'] = [index for index, phase in enumerate(plan.phases, 1) for _ in phase.occurrence_ids]
        groups = schedule.groups.copy()
        groups['phase_id'] = [phase.id for phase in plan.phases]
        groups['query_count'] = [len(phase.occurrence_ids) for phase in plan.phases]
        groups['distinct_query_texts'] = [int(data.loc[list(phase.occurrence_ids)].query_text.nunique()) for phase in plan.phases]
        groups['profile_counts'] = [canonical(data.loc[list(phase.occurrence_ids)].query_type.value_counts().to_dict())
                                    for phase in plan.phases]
        rows = MeasuredCatalog(rows).queries
        return ScenarioResult('GENERATED', SimpleNamespace(rows=rows, groups=groups, schedule=schedule), plan)

    def cohort(self, bag, profile, count, *, excluded=()):
        profiles = set(profile_name(dict(profile=profile), 'profile', multiple=True))
        if profiles - {entry['id'] for entry in json.loads(bag.profiles_json)['profiles']}:
            raise ValueError(f'Unknown execution profile: {profile}')
        entries = [item for item in bag.occurrences if item.profile_id in profiles and item.occurrence_id not in excluded]
        if len(entries) < count:
            return Skip('insufficient_profile_occurrences', dict(profile=profile, required=count, available=len(entries)))
        return entries


class RandomPoissonScenario(ArrivalScenario):
    scenario_type = 'random_poisson'
    fields = ('mu',)

    def validate_config(self):
        PoissonArrivalTime(self.config.get('mu'))

    def plan(self, bag, rng):
        ids = tuple(bag.occurrences[index].occurrence_id for index in rng.permutation(len(bag.occurrences)))
        return ScenarioPlan((ScenarioPhase('poisson', ids, PoissonArrivalTime(self.config['mu'])),), canonical(self.config))


class RepeatedCpuBurstScenario(ArrivalScenario):
    scenario_type = 'repeated_cpu_burst'
    minimum_cohort_counts = (1, 2)
    fields = ('background_profile', 'heavy_profile', 'background_counts', 'burst_count', 'bursts',
              'burst_window_s', 'mu_bg')

    def validate_config(self):
        for key in ('background_profile', 'heavy_profile'):
            profile_name(self.config, key, multiple=True)
        if (set(profile_name(self.config, 'background_profile', multiple=True))
                & set(profile_name(self.config, 'heavy_profile', multiple=True))):
            raise ValueError('Background and heavy profile IDs must be disjoint')
        integer(self.config.get('bursts', 3), 'bursts', 1)
        integer(self.config.get('burst_count'), 'burst_count', 2)
        counts = self.config.get('background_counts')
        if not isinstance(counts, list) or len(counts) != self.config.get('bursts', 3):
            raise ValueError('background_counts must contain one count per burst')
        for count in counts:
            integer(count, 'background count', 1)
        BurstArrivalTime(self.config.get('burst_window_s'))
        PoissonArrivalTime(self.config.get('mu_bg'))

    def plan(self, bag, rng):
        cfg = self.config
        counts = cfg['background_counts']
        heavy_count = cfg['burst_count'] * len(counts)
        background = self.cohort(bag, cfg['background_profile'], sum(counts))
        heavy = self.cohort(bag, cfg['heavy_profile'], heavy_count)
        for entries in (background, heavy):
            if isinstance(entries, Skip):
                return entries
        background = spread_occurrences(background[:sum(counts)], counts, rng)
        heavy = spread_occurrences(heavy[:heavy_count], [cfg['burst_count']] * len(counts), rng)
        phases = []
        for index, (ordinary, burst) in enumerate(zip(background, heavy), 1):
            phases.extend((ScenarioPhase(f'background_{index}', ordinary, PoissonArrivalTime(cfg['mu_bg'])),
                           ScenarioPhase(f'burst_{index}', burst, BurstArrivalTime(cfg['burst_window_s']))))
        return self.finish_plan(bag, phases, rng)


class LongFirstConvoyScenario(ArrivalScenario):
    scenario_type = 'long_first_convoy'
    minimum_cohort_counts = (2, 1)
    fields = ('long_profile', 'fast_profile', 'long_count', 'fast_count', 'long_window_s', 'mu_fast')

    def validate_config(self):
        for key in ('long_profile', 'fast_profile'):
            profile_name(self.config, key, multiple=True)
        if self.config['long_profile'] == self.config['fast_profile']:
            raise ValueError('Long and fast profile IDs must differ')
        integer(self.config.get('long_count'), 'long_count', 2)
        integer(self.config.get('fast_count'), 'fast_count', 1)
        BurstArrivalTime(self.config.get('long_window_s'))
        PoissonArrivalTime(self.config.get('mu_fast'))

    def plan(self, bag, rng):
        cfg = self.config
        phases = []
        used = set()
        for key, policy in [('long', BurstArrivalTime(cfg['long_window_s'])), ('fast', PoissonArrivalTime(cfg['mu_fast']))]:
            count = cfg[f'{key}_count']
            entries = self.cohort(bag, cfg[f'{key}_profile'], count, excluded=used)
            if isinstance(entries, Skip):
                return entries
            phases.append(ScenarioPhase(key, spread_occurrences(entries[:count], [count], rng)[0], policy))
            used.update(phases[-1].occurrence_ids)
        return self.finish_plan(bag, phases, rng)


class PhasedRateScenario(ArrivalScenario):
    scenario_type = 'phased_rate'
    fields = ('phases',)

    def validate_config(self):
        phases = self.config.get('phases')
        if not isinstance(phases, list) or not phases:
            raise ValueError('phases must be a nonempty array')
        for phase in phases:
            object_fields(phase, ('count', 'mu'), 'rate phase')
            integer(phase.get('count'), 'phase count', 1)
            PoissonArrivalTime(phase.get('mu'))

    def plan(self, bag, rng):
        counts = [phase['count'] for phase in self.config['phases']]
        if sum(counts) != len(bag.occurrences):
            return Skip('phase_count_mismatch', dict(required=sum(counts), available=len(bag.occurrences)))
        assignments = spread_occurrences(bag.occurrences, counts, rng)
        phases = tuple(ScenarioPhase(f'rate_{index}', ids, PoissonArrivalTime(phase['mu']))
                       for index, (ids, phase) in enumerate(zip(assignments, self.config['phases']), 1))
        return ScenarioPlan(phases, canonical(self.config))


