"""Group query logs, profile the database, sample workloads and assign arrivals."""
import argparse
import getpass
import json
import math
import os


def positive(value):
    number=float(value)
    if not math.isfinite(number) or number<=0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def nonnegative(value):
    number=float(value)
    if not math.isfinite(number) or number<0:
        raise argparse.ArgumentTypeError('must be nonnegative and finite')
    return number


def positive_int(value):
    number=int(value)
    if number<1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    groups=commands.add_parser('groups',help='Prepare query groups offline; no database requests')
    groups.add_argument('--input',required=True)
    groups.add_argument('--group-size',type=positive_int,required=True)
    groups.add_argument('--sample-size',type=positive_int)
    groups.add_argument('--runtime-min',type=nonnegative)
    groups.add_argument('--runtime-max',type=nonnegative)
    groups.add_argument('--runtime-column')
    groups.add_argument('--seed',type=int,default=42)
    groups.add_argument('--out',required=True)
    profile=commands.add_parser('profile',help='Execute query groups sequentially and retain database CPU/collectl records')
    profile.add_argument('--groups',nargs='+',required=True,help='Group directories or a parent containing groups.json')
    profile.add_argument('--endpoint',required=True)
    transport=profile.add_mutually_exclusive_group()
    transport.add_argument('--ssh-host',help='Database host; defaults to the endpoint hostname')
    transport.add_argument('--local',action='store_true',help='Monitor a database in this process namespace')
    profile.add_argument('--ssh-user',default='root')
    profile.add_argument('--ssh-port',type=positive_int,default=22)
    profile.add_argument('--ssh-identity-file')
    profile.add_argument('--ssh-password-env',default='WRK_GEN_SSH_PASSWORD')
    profile.add_argument('--ssh-ask-password',action='store_true')
    profile.add_argument('--pid',type=positive_int)
    profile.add_argument('--query-timeout',type=positive,default=30.)
    profile.add_argument('--repetitions',type=positive_int,default=1)
    profile.add_argument('--cpu-interval',type=positive,default=.001)
    profile.add_argument('--collectl-interval',type=positive,default=.01)
    profile.add_argument('--cpu-window',type=positive,default=.01)
    profile.add_argument('--query-gap',type=nonnegative,default=.05)
    profile.add_argument('--out',required=True)
    sample=commands.add_parser('sample',help='Freeze a workload bag from a profiled catalog')
    sample.add_argument('--input',nargs='+',required=True,help='One or more profiled CSV files')
    sample.add_argument('--count',type=positive_int)
    sample.add_argument('--recipe',help='JSON execution-profile predicates and ordered group counts')
    sample.add_argument('--exclude',nargs='*',default=[])
    sample.add_argument('--seed',type=int,help='Override the composition sampling seed')
    sample.add_argument('--profiles',help='JSON execution-profile definitions')
    sample.add_argument('--composition',help='JSON dataset/name, counts, reuse and sampling seed')
    sample.add_argument('--group',action='append',default=[],help='PROFILE=COUNT; repeat for each group or override')
    sample.add_argument('--name',help='Workload ID used in occurrence IDs')
    sample.add_argument('--dataset',help='Dataset label')
    sample.add_argument('--reuse',choices=['prefer_distinct','distinct_only'])
    sample.add_argument('--screen-timeouts',action='store_true',help='Exclude observations at recorded collection deadlines')
    sample.add_argument('--allow-reuse',action='store_true')
    sample.add_argument('--runtime-min',type=nonnegative)
    sample.add_argument('--runtime-max',type=nonnegative)
    sample.add_argument('--cpu-min',type=nonnegative)
    sample.add_argument('--cpu-max',type=nonnegative)
    sample.add_argument('--out',required=True)
    arrivals=commands.add_parser('arrivals',help='Assign P1-P4 order and arrivals to an existing bag')
    arrivals.add_argument('--bag',required=True)
    arrivals.add_argument('--pattern',choices=['P1','P2','P3','P4'])
    arrivals.add_argument('--scenario',help='Scenario JSON or a saved workload config.json')
    arrivals.add_argument('--order-seed',type=int)
    arrivals.add_argument('--arrival-seed',type=int)
    arrivals.add_argument('--rate',type=positive)
    arrivals.add_argument('--first-arrival',choices=['initial_gap','zero'])
    arrivals.add_argument('--bursts',type=positive_int)
    arrivals.add_argument('--burst-count',type=positive_int)
    arrivals.add_argument('--burst-window',type=positive)
    arrivals.add_argument('--background-counts',type=positive_int,nargs='+')
    arrivals.add_argument('--heavy-profile',nargs='+')
    arrivals.add_argument('--background-profile',nargs='+')
    arrivals.add_argument('--long-count',type=positive_int)
    arrivals.add_argument('--long-profile',nargs='+')
    arrivals.add_argument('--fast-profile',nargs='+')
    arrivals.add_argument('--fast-count',type=positive_int)
    arrivals.add_argument('--phase-counts',type=positive_int,nargs='+')
    arrivals.add_argument('--phase-rates',type=positive,nargs='+')
    arrivals.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try:
        if any(value is not None and value < 0 for value in (getattr(args,'seed',None),getattr(args,'order_seed',None),getattr(args,'arrival_seed',None))):
            raise ValueError('Seeds must be nonnegative')
        if hasattr(args,'runtime_min') and args.runtime_min is not None and args.runtime_max is not None and args.runtime_min>args.runtime_max:
            raise ValueError('Runtime bounds are reversed')
        if args.command=='groups':
            from .groups import create_groups
            result=create_groups(args.input,args.out,args.group_size,args.sample_size,args.seed,args.runtime_min,args.runtime_max,args.runtime_column)
        elif args.command=='profile':
            from .profiling import profile_groups
            if args.ssh_ask_password:
                os.environ[args.ssh_password_env]=getpass.getpass('Database SSH password: ')
            result=profile_groups(args.groups,args.out,args.endpoint,ssh_host=args.ssh_host,local=args.local,
                                  ssh_user=args.ssh_user,ssh_port=args.ssh_port,ssh_identity_file=args.ssh_identity_file,
                                  ssh_password_env=args.ssh_password_env,pid=args.pid,query_timeout=args.query_timeout,
                                  repetitions=args.repetitions,cpu_interval=args.cpu_interval,collectl_interval=args.collectl_interval,
                                  cpu_window=args.cpu_window,query_gap=args.query_gap)
        elif args.command=='sample':
            from .sampling import sample_workload
            result=sample_workload(args.input,args.out,count=args.count,recipe=args.recipe,seed=args.seed,
                                   exclusions=args.exclude,allow_reuse=args.allow_reuse,profiles=args.profiles,composition=args.composition,
                                   group_counts=args.group,name=args.name,dataset=args.dataset,reuse=args.reuse,
                                   screen_timeouts=args.screen_timeouts,runtime_min=args.runtime_min,
                                   runtime_max=args.runtime_max,cpu_min=args.cpu_min,cpu_max=args.cpu_max)
        else:
            from .arrivals import generate_arrivals
            result=generate_arrivals(args)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError,OSError,RuntimeError,KeyError) as exc:
        parser.exit(1,f'wrk_gen: {exc}\n')
