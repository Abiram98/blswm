"""Immutable occurrence bags and execution-profile composition validation."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import pandas as pd
from .queries import new_directory, fingerprint, write_json
from .profiles import parse_profiles, integer, object_fields

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def file_manifest(root):
    return {str(path.relative_to(root)): fingerprint(path) for path in sorted(root.rglob('*'))
            if path.is_file() and path != root / 'files_sha256.json'}


@dataclass(frozen=True)
class QueryOccurrence:
    occurrence_id: str
    source_catalog: str
    source_query_id: str
    query_text: str
    profile_id: str
    metrics_json: str

    def row(self):
        return dict(json.loads(self.metrics_json), query_id=self.occurrence_id,
                    source_query_id=self.source_query_id, source_catalog=self.source_catalog,
                    query_text=self.query_text, query_type=self.profile_id, stratum=self.profile_id,
                    query_sha256=sha256(self.query_text.encode()).hexdigest())


@dataclass(frozen=True)
class QueryBag:
    id: str
    occurrences: tuple[QueryOccurrence, ...]
    profiles_json: str
    recipe_json: str
    provenance_json: str

    def payload(self):
        return dict(schema_version=1, occurrences=[asdict(item) for item in self.occurrences],
                    profiles=json.loads(self.profiles_json), recipe=json.loads(self.recipe_json),
                    provenance=json.loads(self.provenance_json))

    @classmethod
    def create(cls, occurrences, profiles, recipe, provenance):
        bag = cls('', tuple(occurrences), canonical(profiles), canonical(recipe), canonical(provenance))
        return cls(digest(bag.payload()), bag.occurrences, bag.profiles_json, bag.recipe_json, bag.provenance_json)

    def frame(self):
        return pd.DataFrame([item.row() for item in self.occurrences])

    def validate(self):
        if not self.occurrences or self.id != digest(self.payload()):
            raise ValueError('Empty or altered query bag')
        data = self.frame()
        if data.query_id.duplicated().any() or data.query_text.str.strip().eq('').any():
            raise ValueError('Bag requires unique occurrence IDs and nonempty query texts')
        profiles = {profile.id: profile for profile in parse_profiles(json.loads(self.profiles_json))}
        recipe = json.loads(self.recipe_json)
        validate_composition(recipe, profiles)
        if data.query_type.value_counts().to_dict() != {key: count for key, count in recipe['counts'].items() if count}:
            raise ValueError('Bag profile counts differ from composition recipe')
        for key, group in data.groupby('query_type'):
            if key not in profiles or not profiles[key].eligible(group).all():
                raise ValueError(f'Bag violates execution profile: {key}')
        if recipe.get('reuse', 'prefer_distinct') == 'distinct_only' and data.query_text.duplicated().any():
            raise ValueError('Bag violates distinct-only sampling')

    def save(self, destination):
        self.validate()
        destination = new_directory(destination)
        write_json(destination / 'bag.json', dict(bag_id=self.id, **self.payload()))
        data = self.frame()
        data.to_csv(destination / 'occurrences.csv', index=False, float_format='%.17g')
        write_json(destination / 'profiles.json', json.loads(self.profiles_json))
        write_json(destination / 'composition.json', json.loads(self.recipe_json))
        write_json(destination / 'validation.json', dict(state='complete', bag_id=self.id,
                   occurrences=len(data), distinct_query_texts=int(data.query_text.nunique())))
        write_json(destination / 'files_sha256.json', file_manifest(destination))
        return destination

    @classmethod
    def load(cls, path):
        path = Path(path)
        path = path / 'bag.json' if path.is_dir() else path
        value = json.loads(path.read_text())
        if value.get('schema_version') != 1:
            raise ValueError('Unsupported bag schema_version')
        bag = cls(value['bag_id'], tuple(QueryOccurrence(**item) for item in value['occurrences']),
                  canonical(value['profiles']), canonical(value['recipe']), canonical(value['provenance']))
        bag.validate()
        return bag


def validate_composition(recipe, profiles):
    object_fields(recipe, ('id', 'dataset', 'counts', 'sampling_seed', 'reuse'), 'composition')
    if not isinstance(recipe.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', recipe['id']):
        raise ValueError('Composition needs a safe nonempty id')
    if not isinstance(recipe.get('dataset'), str) or not recipe['dataset'].strip():
        raise ValueError('Composition needs an explicit dataset label')
    integer(recipe.get('sampling_seed'), 'sampling_seed')
    if recipe.get('reuse', 'prefer_distinct') not in ('prefer_distinct', 'distinct_only'):
        raise ValueError('reuse must be prefer_distinct or distinct_only')
    counts = recipe.get('counts')
    if not isinstance(counts, dict) or not counts or counts.keys() - profiles.keys():
        raise ValueError('counts must reference known execution profiles')
    for count in counts.values():
        integer(count, 'profile count')
    if not sum(counts.values()):
        raise ValueError('Composition must be nonempty')


