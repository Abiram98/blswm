"""Configure and export P1-P4 scenarios over immutable occurrence bags."""
import json
from pathlib import Path
from .bag import QueryBag, file_manifest
from .queries import new_directory, fingerprint, write_json
from .scenarios import (RandomPoissonScenario, RepeatedCpuBurstScenario,
                        LongFirstConvoyScenario, PhasedRateScenario)

SCENARIOS = {'P1': RandomPoissonScenario, 'P2': RepeatedCpuBurstScenario,
             'P3': LongFirstConvoyScenario, 'P4': PhasedRateScenario}


def distribute(total, groups):
    if total < groups:
        raise ValueError('Every background phase needs at least one query')
    result = [total // groups] * groups
    for index in range(total % groups):
        result[groups - 1 - index] += 1
    return result


def scenario_config(args, bag):
    supplied = {}
    if args.scenario:
        supplied = json.loads(Path(args.scenario).read_text())
        supplied = supplied.get('scenario', supplied)
    by_type = {cls.scenario_type: key for key, cls in SCENARIOS.items()}
    pattern = args.pattern or by_type.get(supplied.get('type'))
    if pattern not in SCENARIOS:
        raise ValueError('Specify --pattern or a P1-P4 --scenario JSON file')
    if supplied and supplied.get('type') != SCENARIOS[pattern].scenario_type:
        raise ValueError('--pattern conflicts with the supplied scenario type')
    n = len(bag.occurrences)
    known = [profile['id'] for profile in json.loads(bag.profiles_json)['profiles']]
    config = dict(id=pattern, type=SCENARIOS[pattern].scenario_type, order_seed=202,
                  arrival_seed=303, first_arrival_rule='initial_gap', transition='previous_group_policy')
    config.update(supplied)
    for argument, key in [('order_seed','order_seed'), ('arrival_seed','arrival_seed'),
                          ('first_arrival','first_arrival_rule')]:
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    def option(argument, key, default):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
        elif key not in config:
            config[key] = default
    if pattern == 'P1':
        option('rate', 'mu', .6)
    elif pattern == 'P2':
        option('heavy_profile', 'heavy_profile', 'H_M')
        option('bursts', 'bursts', 3)
        option('burst_count', 'burst_count', 10)
        option('burst_window', 'burst_window_s', .5)
        option('rate', 'mu_bg', .6)
        heavy = config['heavy_profile']
        heavy = [heavy] if isinstance(heavy, str) else heavy
        option('background_profile', 'background_profile', [key for key in known if key not in heavy])
        if args.background_counts is not None:
            config['background_counts'] = args.background_counts
        elif 'background_counts' not in config:
            config['background_counts'] = distribute(n - config['bursts'] * config['burst_count'], config['bursts'])
    elif pattern == 'P3':
        option('long_profile', 'long_profile', 'V100')
        option('fast_profile', 'fast_profile', known)
        option('long_count', 'long_count', 10)
        option('fast_count', 'fast_count', n - config['long_count'])
        option('burst_window', 'long_window_s', .5)
        option('rate', 'mu_fast', .6)
    else:
        if args.phase_counts is not None or args.phase_rates is not None or 'phases' not in config:
            previous = config.get('phases', [])
            counts = args.phase_counts or ([phase['count'] for phase in previous] if previous else distribute(n, 3))
            rates = args.phase_rates or ([phase['mu'] for phase in previous] if previous else [.2, 1., .2])
            if len(counts) != len(rates):
                raise ValueError('Supply one phase rate for each phase count')
            config['phases'] = [dict(count=count, mu=rate) for count, rate in zip(counts, rates)]
    return pattern, config


def generate_arrivals(args):
    source = Path(args.bag).expanduser().resolve()
    source = source / 'bag.json' if source.is_dir() else source
    source_hash = fingerprint(source)
    bag = QueryBag.load(source)
    scenario_hash = fingerprint(args.scenario) if args.scenario else None
    pattern, config = scenario_config(args, bag)
    result = SCENARIOS[pattern](config).generate(bag)
    if result.status != 'GENERATED':
        raise ValueError(f'{result.reason}: {result.details}')
    workload = result.workload
    rows, groups, schedule = workload.rows, workload.groups, workload.schedule
    destination = new_directory(args.out, [source.parent])
    rows.to_csv(destination / 'sampled_measurements.csv', index=False, float_format='%.17g')
    rows[['query_id', 'query_text', 'runtime_seconds']].rename(columns={
        'query_id': 'id', 'query_text': 'text', 'runtime_seconds': 'ex time (secs)'
    }).to_csv(destination / 'ql.csv', index=False, float_format='%.17g')
    schedule.arrivals.to_csv(destination / 'arrival_time.csv', index=False, float_format='%.17g')
    groups.to_csv(destination / 'groups.csv', index=False)
    write_json(destination / 'config.json', dict(composition=json.loads(bag.recipe_json), scenario=config, bag_id=bag.id))
    write_json(destination / 'bag.json', dict(bag_id=bag.id, **bag.payload()))
    write_json(destination / 'profiles.json', json.loads(bag.profiles_json))
    times = schedule.arrivals.arrival_time
    write_json(destination / 'validation.json', dict(state='complete', bag_id=bag.id,
        occurrences=len(rows), distinct_query_texts=int(rows.query_text.nunique()),
        frozen_occurrences_preserved=True, profile_counts=rows.query_type.value_counts().to_dict()))
    write_json(destination / 'manifest.json', dict(schema_version=1, bag_id=bag.id,
        bag_sha256=source_hash, scenario_sha256=scenario_hash, pattern=pattern,
        queries=len(rows), config=config, transition='previous_group_policy',
        effective_arrival_rate=schedule.effective_rate, horizon_seconds=float(times.iloc[-1])))
    if fingerprint(source) != source_hash or args.scenario and fingerprint(args.scenario) != scenario_hash:
        raise ValueError('Frozen inputs changed during arrival generation')
    write_json(destination / 'files_sha256.json', file_manifest(destination))
    return dict(output=str(destination), pattern=pattern, queries=len(rows),
                horizon_seconds=float(times.iloc[-1]), phases=len(groups))
