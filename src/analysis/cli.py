"""Prepare execution logs, compare workload runs, and analyze prediction quality."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare',help='Prepare raw execution runs without modifying them')
    prep.add_argument('--runs',nargs='+',required=True,help='Run directories or directories to search recursively')
    prep.add_argument('--time-base',choices=['epoch','perf_counter'],default='epoch',help='Legacy timing clock; epoch resource timestamps are interpreted as UTC')
    prep.add_argument('--resource-interval',type=float,default=.1,help='Seconds per resource sample for perf_counter timing')
    prep.add_argument('--out',required=True)
    workload=commands.add_parser('workloads',help='Analyze named WorkloadRunConfig groups')
    workload.add_argument('--prepared',required=True)
    groups=workload.add_mutually_exclusive_group(required=True)
    groups.add_argument('--config',help='JSON containing configs with display_name and run_path')
    groups.add_argument('--run',action='append',nargs='+',metavar=('NAME','PATH'),help='Policy name followed by one or more run paths; repeat for each policy')
    workload.add_argument('--metric',choices=['latency_s','execution_s','wait_s'],default='latency_s')
    workload.add_argument('--reference')
    workload.add_argument('--include-incomplete',action='store_true')
    workload.add_argument('--out',required=True)
    quality=commands.add_parser('predictions',help='Analyze classification or regression predictions')
    quality.add_argument('--data',required=True,help='Labeled query-set JSONL or candidate CSV/TSV')
    group=quality.add_mutually_exclusive_group(required=True)
    group.add_argument('--report',help='qpp evaluate JSON report')
    group.add_argument('--port',type=int,help='TCP prediction server port')
    quality.add_argument('--host',default='127.0.0.1')
    quality.add_argument('--timeout',type=float,default=60.)
    quality.add_argument('--thresholds',type=float,nargs='+')
    quality.add_argument('--out',required=True)
    compare=commands.add_parser('compare-predictions',help='Compare models on identical candidate observations')
    compare.add_argument('--results',nargs='+',required=True,help='Prediction analysis directories')
    compare.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            from .api import PreparationConfig
            result=PreparationConfig(args.runs,args.out,args.time_base,args.resource_interval).run()
        elif args.command=='workloads':
            from .api import WorkloadAnalysisConfig,WorkloadRunConfig
            if args.run:
                if any(len(group)<2 for group in args.run):raise ValueError('Each --run needs a name and at least one path')
                configs=[WorkloadRunConfig(group[1:],group[0]) for group in args.run]
            else:
                config_path=Path(args.config).resolve(); payload=json.loads(config_path.read_text())
                raw=payload.get('configs') if isinstance(payload,dict) else payload
                if not isinstance(raw,list) or not raw:raise ValueError('Config must contain a nonempty configs array')
                configs=[]
                for entry in raw:
                    if not isinstance(entry,dict) or set(entry)-{'display_name','run_path'} or 'run_path' not in entry:
                        raise ValueError('Each config needs run_path and optional display_name')
                    paths=entry['run_path'] if isinstance(entry['run_path'],list) else [entry['run_path']]
                    paths=[str((config_path.parent/p).resolve()) if not Path(p).is_absolute() else p for p in paths]
                    configs.append(WorkloadRunConfig(paths,entry.get('display_name')))
            result=WorkloadAnalysisConfig(args.prepared,configs,args.out,args.metric,args.reference,args.include_incomplete).run()
        elif args.command=='compare-predictions':
            from .api import PredictionComparisonConfig
            result=PredictionComparisonConfig(args.results,args.out).run()
        else:
            from .api import PredictionAnalysisConfig
            result=PredictionAnalysisConfig(args.data,args.out,args.report,args.host,args.port,args.timeout,args.thresholds).run()
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except KeyboardInterrupt:return 130
    except (OSError,ValueError,RuntimeError,KeyError) as exc:parser.exit(1,f'analysis: {exc}\n')
