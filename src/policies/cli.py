"""Replay a saved arrival workload with classification or runtime-based policies."""
import argparse
import json
import math
import sys
from pathlib import Path
from .scheduling import Policy
from .runtime import HTTPExecutor, replay
from .schema import FakePredictor
from .workload import load_workload
from qpp.artifacts import output_path

NAMES = ('fifo', 'rtq', 'rtq-reeval', 'sqa', 'sqa-reeval', 'rr', 'iconqshed', 'iconq')


def configure_parser(parser, policy=None):
    """Share execution arguments while exposing each policy's own controls."""
    parser.set_defaults(policy=policy or 'fifo', class_capacities=[5, 5], no_spillover=False,
                        budget=10., class_costs=[1., 3.], threshold_seconds=120.,
                        predictor_host='127.0.0.1', predictor_port=9010, predictor_timeout=60.)
    parser.add_argument('--workload-dir', required=True)
    parser.add_argument('--arrival-time-csv')
    if policy is None:
        parser.add_argument('--policy', choices=NAMES, default='fifo')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--out', required=True, help='New directory for results')
    parser.add_argument('--mpl', type=int, default=10)
    if policy is None or policy.startswith('rtq'):
        parser.add_argument('--class-capacities', type=int, nargs=2, default=[5, 5])
        parser.add_argument('--no-spillover', action='store_true')
    if policy is None or policy.startswith('sqa'):
        parser.add_argument('--budget', type=float, default=10)
        parser.add_argument('--class-costs', type=float, nargs=2, default=[1, 3])
    if policy is None or policy in ('rtq', 'rtq-reeval', 'sqa', 'sqa-reeval', 'rr'):
        parser.add_argument('--threshold-seconds', type=float, default=120)
    if policy != 'fifo':
        parser.add_argument('--predictor-host', default='127.0.0.1')
        parser.add_argument('--predictor-port', type=int, default=9011 if policy in ('iconqshed', 'iconq') else 9010)
        parser.add_argument('--predictor-timeout', type=float, default=60)
    parser.add_argument('--query-timeout', type=float, default=300)
    if policy is None or policy in ('iconqshed', 'iconq'):
        parser.add_argument('--alpha', type=float, default=.2)
        parser.add_argument('--starve-penalty', type=float, default=.5)
        parser.add_argument('--steps-into-future', type=int, default=2)
        parser.add_argument('--ignore-short-running', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--short-running-threshold', type=float, default=5.)
        parser.add_argument('--future-wait-floor', type=float, default=3.)
        parser.add_argument('--overrun-extension-factor', type=float, default=.15)
        parser.add_argument('--reconsider-interval', type=float, default=5.)
    parser.add_argument('--run-id', default='run1')
    parser.add_argument('--limit', type=int, help='Explicit smoke-test prefix (all inputs are validated first)')
    parser.add_argument('--dry-run', action='store_true', help='Validate and preview without networking or writes')


def parse_args(argv):
    # Retain the original flag-based entry for existing scripts.
    if argv and argv[0].startswith('-') and argv[0] not in ('-h', '--help'):
        parser = argparse.ArgumentParser(description=__doc__)
        configure_parser(parser)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest='policy', required=True)
        descriptions = {
            'fifo': 'FIFO admission without a predictor',
            'rtq': 'Runtime-class pools with configurable capacities',
            'rtq-reeval': 'RTQ with running-query reclassification',
            'sqa': 'Class-cost admission with a fixed running budget',
            'sqa-reeval': 'SQA with updated running classes and costs',
            'rr': 'Round robin between fast and slow query queues',
            'iconqshed': 'Runtime-regression lookahead scheduling',
        }
        for name, description in descriptions.items():
            aliases = ['iconq'] if name == 'iconqshed' else []
            command = commands.add_parser(name, aliases=aliases, help=description, description=description)
            configure_parser(command, name)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if not math.isfinite(args.threshold_seconds) or args.threshold_seconds <= 0:
            raise ValueError('threshold-seconds must be positive and finite')
        if not math.isfinite(args.query_timeout) or args.query_timeout <= 0:
            raise ValueError('query-timeout must be positive and finite')
        if not math.isfinite(args.predictor_timeout) or args.predictor_timeout <= 0:
            raise ValueError('predictor-timeout must be positive and finite')
        trace, catalog = load_workload(args.workload_dir, args.arrival_time_csv)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError('limit must be positive')
            trace = trace[:args.limit]
        options = {}
        if args.policy in ('iconq', 'iconqshed'):
            options = {key: getattr(args, key) for key in (
                'alpha', 'starve_penalty', 'steps_into_future', 'ignore_short_running',
                'short_running_threshold', 'future_wait_floor', 'overrun_extension_factor', 'reconsider_interval')}
        policy = Policy(args.policy, args.mpl, args.budget, class_capacities=args.class_capacities,
                        class_costs=args.class_costs, spillover=not args.no_spillover, **options)
        destination = output_path(args.out)
        workload = Path(args.workload_dir).resolve()
        if destination == workload or destination.is_relative_to(workload):
            raise ValueError('Results must be outside the workload directory')
        if destination.exists():
            raise ValueError(f'Result directory already exists: {destination}')
        if args.dry_run:
            print(json.dumps(dict(policy=args.policy, occurrences=len(trace), endpoint=args.endpoint,
                                  output=str(destination), requires_predictor=args.policy != 'fifo')))
            return 0
        predictor = FakePredictor()
        if args.policy != 'fifo':
            from qpp.client import RemotePredictor
            predictor = RemotePredictor(catalog, args.predictor_host, args.predictor_port,
                                        args.predictor_timeout, (args.threshold_seconds,), task=policy.prediction_task)
        destination.mkdir(parents=True, exist_ok=False)
        result = replay(trace, catalog, policy, predictor,
                        executor=HTTPExecutor(args.endpoint, args.query_timeout), run_id=args.run_id)
        result['configuration'] = vars(args)
        if args.policy != 'fifo':
            result['predictor_info'] = predictor.metadata
        (destination / 'run.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        for name in ('records', 'predictions', 'decisions', 'running_updates'):
            (destination / f'{name}.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in result[name]))
        print(json.dumps(dict(output=str(destination), complete=result['complete'],
                              occurrences=len(result['records']), controller_error=result['controller_error'])))
        return 0 if result['complete'] else 1
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'policies: {exc}\n')
