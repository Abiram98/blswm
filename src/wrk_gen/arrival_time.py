"""Seeded group arrival policies and explicit transitions between query groups."""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def burst_offsets(count, window, rng):
    """The CPU-burst generator's fixed endpoints and sorted uniform interior."""
    if type(count) is not int or count < 2:
        raise ValueError('BurstArrivalTime needs at least two queries per group')
    positive(window, 'arrival_window_seconds')
    return np.r_[0., np.sort(rng.uniform(0, window, count - 2)), window]


@dataclass(frozen=True)
class PoissonArrivalTime:
    arrival_rate: float = .6
    type: str = field(default='PoissonArrivalTime', init=False)

    def __post_init__(self):
        positive(self.arrival_rate, 'arrival_rate')

    def internal_gaps(self, count, rng):
        return rng.exponential(1. / self.arrival_rate, count - 1)

    def next_gap(self, count, rng):
        return float(rng.exponential(1. / self.arrival_rate))


@dataclass(frozen=True)
class BurstArrivalTime:
    arrival_window_seconds: float = 1.
    type: str = field(default='BurstArrivalTime', init=False)

    def __post_init__(self):
        positive(self.arrival_window_seconds, 'arrival_window_seconds')

    def internal_gaps(self, count, rng):
        return np.diff(burst_offsets(count, self.arrival_window_seconds, rng))

    def next_gap(self, count, rng):
        # A fresh first spacing has the same law as any spacing within this burst.
        return float(burst_offsets(count, self.arrival_window_seconds, rng)[1])


def arrival_from_dict(value):
    if not isinstance(value, dict):
        raise ValueError('arrival must be an object with a type')
    options = dict(value)
    kind = options.pop('type', None)
    cls = {'PoissonArrivalTime': PoissonArrivalTime, 'PoisonArrivalTime': PoissonArrivalTime,
           'BurstArrivalTime': BurstArrivalTime}.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ValueError('arrival.type must be PoissonArrivalTime or BurstArrivalTime')
    allowed = {'arrival_rate'} if cls is PoissonArrivalTime else {'arrival_window_seconds'}
    if options.keys() - allowed:
        raise ValueError(f'{kind} supports only {", ".join(sorted(allowed))}')
    return cls(**options)


@dataclass
class GroupArrivalSchedule:
    arrivals: pd.DataFrame
    groups: pd.DataFrame
    folder: str
    effective_rate: float


class GroupArrivalGenerator:
    """Each group's first gap belongs to the previous policy; the first uses its own."""

    def __init__(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Arrival seed must be a nonnegative integer')
        self.seed = seed

    def generate(self, groups, *, default_rate=.6, first_arrival_rule='initial_gap'):
        if not groups:
            raise ValueError('An arrival schedule needs at least one group')
        if first_arrival_rule not in ('initial_gap', 'zero'):
            raise ValueError('first_arrival_rule must be initial_gap or zero')
        policies = [group.arrival or PoissonArrivalTime(default_rate) for group in groups]
        rng = np.random.default_rng(self.seed)
        count = sum(group.query_count for group in groups)
        gaps = np.empty(count, dtype=float)
        records, cursor = [], 0
        for index, (group, policy) in enumerate(zip(groups, policies)):
            source = max(0, index - 1)
            gaps[cursor] = (0. if index == 0 and first_arrival_rule == 'zero'
                            else policies[source].next_gap(groups[source].query_count, rng))
            stop = cursor + group.query_count
            gaps[cursor + 1:stop] = policy.internal_gaps(group.query_count, rng)
            records.append(dict(group_index=index + 1, first_position=cursor + 1, last_position=stop,
                                arrival_type=policy.type,
                                arrival_rate=getattr(policy, 'arrival_rate', None),
                                arrival_window_seconds=getattr(policy, 'arrival_window_seconds', None),
                                entry_gap_source_group=source + 1, entry_arrival_type=policies[source].type,
                                entry_gap_seconds=float(gaps[cursor])))
            cursor = stop
        times = np.cumsum(gaps)
        if (not np.isfinite(times).all() or not (gaps[1:] > 0).all() or gaps[0] < 0
                or first_arrival_rule == 'initial_gap' and gaps[0] == 0
                or not (times >= 0).all()
                or not (np.diff(times) > 0).all()):
            raise ValueError('Arrival settings cannot produce finite, strictly increasing timestamps at this precision')
        for record in records:
            first, last = times[record['first_position'] - 1], times[record['last_position'] - 1]
            record.update(first_arrival_seconds=float(first), last_arrival_seconds=float(last),
                          arrival_span_seconds=float(last - first))
        effective_rate = float((count - 1) / (times[-1] - times[0]) if count > 1
                               else 1. / times[0] if times[0] > 0 else policies[0].arrival_rate)
        if not math.isfinite(effective_rate) or effective_rate <= 0:
            raise ValueError('Arrival settings produce an unrepresentable effective rate')
        common_rate = policies[0].arrival_rate if all(isinstance(policy, PoissonArrivalTime) for policy in policies) else None
        uniform_rate = common_rate is not None and all(policy.arrival_rate == common_rate for policy in policies)
        rate = common_rate if uniform_rate else effective_rate
        folder = f'arrival_rate_mu_{rate:g}'.replace('.', '_')
        return GroupArrivalSchedule(pd.DataFrame(dict(query_no=np.arange(1, count + 1), arrival_time=times)),
                                    pd.DataFrame(records), folder, effective_rate)
