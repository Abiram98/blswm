"""Data-driven execution profiles; no dataset paths or experimental defaults."""
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

import math

def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


METRICS = ('runtime_seconds', 'peak_cpu_pct', 'mean_cpu_pct')


def object_fields(value, allowed, label):
    if not isinstance(value, dict) or value.keys() - set(allowed):
        raise ValueError(f'Unknown or invalid {label} fields')


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')


@dataclass(frozen=True)
class ExecutionProfile:
    id: str
    predicates_json: str

    @classmethod
    def from_dict(cls, value):
        object_fields(value, ('id', *METRICS), 'execution profile')
        if not isinstance(value.get('id'), str) or not value['id'].strip():
            raise ValueError('Execution profiles need nonempty IDs')
        predicates = {'runtime_seconds': {'min': 0}, 'peak_cpu_pct': {'min': 0, 'max': 3200}}
        for metric in METRICS:
            if metric not in value:
                continue
            bounds = value[metric]
            object_fields(bounds, ('min', 'max', 'min_inclusive', 'max_inclusive', 'unknown'), metric)
            bounds = dict(bounds)
            for key in ('min', 'max'):
                if bounds.get(key) is not None:
                    number(bounds[key], f'{metric}.{key}')
            for key in ('min_inclusive', 'max_inclusive'):
                if key in bounds and type(bounds[key]) is not bool:
                    raise ValueError(f'{metric}.{key} must be boolean')
            if bounds.get('unknown', 'exclude') not in ('include', 'exclude', 'only'):
                raise ValueError(f'{metric}.unknown must be include, exclude or only')
            lower, upper = bounds.get('min'), bounds.get('max')
            if lower is not None and upper is not None and (lower > upper or
                    lower == upper and not (bounds.get('min_inclusive', True) and bounds.get('max_inclusive', True))):
                raise ValueError(f'{metric} bounds describe an empty range')
            predicates[metric] = bounds
        return cls(value['id'], json.dumps(predicates, sort_keys=True, allow_nan=False))

    def to_dict(self):
        return dict(id=self.id, **json.loads(self.predicates_json))

    def eligible(self, data):
        predicates = json.loads(self.predicates_json)
        mask = pd.Series(True, index=data.index)
        for metric, bounds in predicates.items():
            values = data[metric] if metric in data else pd.Series(float('nan'), index=data.index)
            selected = values.notna()
            for key, inclusive, op in [('min', 'min_inclusive', 'ge'), ('max', 'max_inclusive', 'le')]:
                if bounds.get(key) is not None:
                    method = op if bounds.get(inclusive, True) else {'ge': 'gt', 'le': 'lt'}[op]
                    selected &= getattr(values, method)(bounds[key])
            if bounds.get('unknown', 'exclude') == 'include':
                selected |= values.isna()
            elif bounds.get('unknown') == 'only':
                selected = values.isna()
            mask &= selected
        return mask


def parse_profiles(value):
    object_fields(value, ('profiles',), 'profiles file')
    entries = value.get('profiles')
    if not isinstance(entries, list) or not entries:
        raise ValueError('profiles must be a nonempty array')
    profiles = tuple(ExecutionProfile.from_dict(entry) for entry in entries)
    if len({profile.id for profile in profiles}) != len(profiles):
        raise ValueError('Execution profile IDs must be unique')
    return profiles


def load_profiles(path):
    return parse_profiles(json.loads(Path(path).read_text()))
