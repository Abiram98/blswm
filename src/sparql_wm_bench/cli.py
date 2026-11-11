import argparse
import json
from pathlib import Path
from .io import read, write

def main(argv=None):
    parser=argparse.ArgumentParser(description='Standalone benchmark for learned, engine-agnostic workload management (SPARQL workloads)')
    sub=parser.add_subparsers(dest='command',required=True)
    def arguments(p, fields):
        for field in fields: p.add_argument('--'+field,required=True)
    cat=sub.add_parser('catalog'); cat.add_argument('action',choices=['validate']); arguments(cat,['input'])
    w=sub.add_parser('workload'); w.add_argument('action',choices=['generate']); arguments(w,['config','catalog','out'])
    w.add_argument('--exclude')
    r=sub.add_parser('run'); arguments(r,['trace','catalog','policy','out']); r.add_argument('--config'); r.add_argument('--endpoint'); r.add_argument('--synthetic-durations'); r.add_argument('--model-dir'); r.add_argument('--graph-config'); r.add_argument('--run-id',default='run1'); r.add_argument('--mpl',type=int,default=10)
    t=sub.add_parser('trace'); t.add_argument('action',choices=['build']); arguments(t,['runs','catalog','out']); t.add_argument('--test-runs'); t.add_argument('--workload-id',default='W1'); t.add_argument('--mpl',type=int,default=10)
    m=sub.add_parser('model'); m.add_argument('action',choices=['train','evaluate','serve','benchmark']); arguments(m,['graph-config']); m.add_argument('--model-dir'); m.add_argument('--config'); m.add_argument('--train'); m.add_argument('--validation'); m.add_argument('--test'); m.add_argument('--out'); m.add_argument('--port',type=int,default=19022)
    a=sub.add_parser('analyze'); arguments(a,['runs','out']); a.add_argument('--reference',default='fifo_mpl10'); a.add_argument('--plot',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='catalog':
            from .catalog import load
            rows=load(args.input); print(json.dumps(dict(count=len(rows),ineligible=sum(r['profile'] is None for r in rows))))
        elif args.command=='workload':
            from .catalog import load
            from .workloads import select,generate,save
            config=read(args.config)
            if (config.get('composition') or config.get('workload_id') in ('W2','D2')) and not args.exclude:
                raise ValueError('This experiment preset requires --exclude with the training-workload bag')
            bag=select(load(args.catalog),config,read(args.exclude) if args.exclude else ())
            trace=generate(bag,config.get('pattern','P'),config.get('arrival_seed',1),config.get('mu',.6)); save(args.out,bag,trace,config)
        elif args.command=='run':
            from .schema import FakePredictor
            from .policies import Policy
            from .runtime import replay,HTTPExecutor
            catalog=read(args.catalog); config=read(args.config) if args.config else {}
            config=config.get(args.policy,config); config.setdefault('mpl',args.mpl)
            predictor=FakePredictor(default=1 if args.policy=='iconq' else 0)
            if args.model_dir:
                from .ml import Loaded,Online
                from .graphs import Graphs
                predictor=Online(Loaded(args.model_dir,Graphs(read(args.graph_config))),catalog)
                kind=predictor.loaded.config['model_type']
                allowed={'rtq':('planrgcn_class','scqpp_class','bi_scqpp_class'),'sqa':('scqpp_class','bi_scqpp_class'),'rtq-reeval':('bi_scqpp_class',),'sqa-reeval':('bi_scqpp_class',),'iconq':('bi_scqpp_reg',),'rr':('planrgcn_class','scqpp_class','bi_scqpp_class')}
                if args.policy in allowed and kind not in allowed[args.policy]: raise ValueError('Incompatible predictor for configured policy')
            elif not args.synthetic_durations and args.policy!='fifo':
                raise ValueError('Real predictive policies require --model-dir; no implicit mock')
            run=replay(read(args.trace),catalog,Policy(args.policy,**config),predictor,executor=HTTPExecutor(args.endpoint) if args.endpoint else None,
                       durations=read(args.synthetic_durations) if args.synthetic_durations else None,run_id=args.run_id)
            if args.model_dir:
                run['predictor']=predictor.loaded.config['model_type']
                run['setting_id']+='_'+run['predictor']
            write(f'{args.out}/run.json',run); write(f'{args.out}/records.csv',run['records'])
        elif args.command=='trace':
            from .dataset import build,split
            catalog=read(args.catalog)
            samples=sum((build(read(p),catalog,args.workload_id,args.mpl) for p in sorted(Path(args.runs).rglob('run.json'))),[])
            test=sum((build(read(p),catalog,'test',args.mpl) for p in sorted(Path(args.test_runs).rglob('run.json'))),[]) if args.test_runs else []
            train,val,report=split(samples,test)
            for name,data in [('train',train),('validation',val),('test',test)]: write(f'{args.out}/{name}.jsonl',data)
            write(f'{args.out}/leakage.json',report)
        elif args.command=='model':
            from .graphs import Graphs
            from .ml import fit,Loaded,evaluate
            graphs=Graphs(read(args.graph_config))
            if args.action=='train': fit(read(args.config),read(args.train),read(args.validation),graphs,args.out)
            elif args.action=='evaluate': write(f'{args.out}/metrics.json',evaluate(Loaded(args.model_dir,graphs),read(args.test)))
            elif args.action=='benchmark':
                from .ml import benchmark
                items=[dict(x,occurrence_id=f'timing{i}') for i,x in enumerate(read(args.test))]
                write(f'{args.out}/timing.json',benchmark(Loaded(args.model_dir,graphs),items))
            else:
                from .service import serve
                serve(Loaded(args.model_dir,graphs),port=args.port)
        else:
            from .analysis import averaged,compare,summarize,plot,diagnostics
            runs=[read(p) for p in sorted(Path(args.runs).rglob('run.json'))]
            if not runs: raise ValueError('No run.json inputs found')
            if len({r.get('catalog_hash') for r in runs})!=1 or any(r['trace']!=runs[0]['trace'] for r in runs):
                raise ValueError('Query texts or traces differ between policies')
            groups={name:[r for r in runs if r['setting_id']==name] for name in {r['setting_id'] for r in runs}}
            values={name:averaged(rs) for name,rs in groups.items()}
            reference=values[args.reference]
            write(f'{args.out}/comparison.json',{name:compare(v,reference) for name,v in values.items()})
            write(f'{args.out}/summary.json',{name:[summarize(r['records']) for r in rs] for name,rs in groups.items()})
            write(f'{args.out}/means.csv',[dict(policy=name,occurrence_id=k,latency_s=v) for name,vs in values.items() for k,v in vs.items()])
            write(f'{args.out}/diagnostics.json',[diagnostics(r) for r in runs])
            if args.plot: plot(values,f'{args.out}/latency.png')
    except (ValueError,KeyError,TypeError) as e: parser.error(str(e))

if __name__=='__main__': main()
