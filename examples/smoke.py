"""Run with python examples/smoke.py from repository root; no KG/services."""
from sparql_wm_bench.io import write
from sparql_wm_bench.workloads import generate
from sparql_wm_bench.runtime import replay
from sparql_wm_bench.policies import Policy
from sparql_wm_bench.schema import FakePredictor
from sparql_wm_bench.analysis import averaged,compare,summarize
bag=[dict(query_id=f'q{i}',query_text=f'SELECT * WHERE {{ ?s <urn:p{i}> ?o }}',profile='M') for i in range(350)]
trace=generate(bag,'P',seed=12)
write('runs/smoke/inputs/catalog.json',bag); write('runs/smoke/inputs/trace.csv',trace)
all_runs={}
for name in ('fifo','rtq','sqa','iconq'):
    repeats=[]
    for rep in range(2):
        run=replay(trace,bag,Policy(name),FakePredictor(default=2 if name=='iconq' else 0),durations={q['query_id']:2 for q in bag},run_id=f'{name}_{rep}')
        write(f'runs/smoke/{name}_{rep}/run.json',run); repeats.append(run)
    all_runs[name]=averaged(repeats)
write('runs/smoke/comparison.json',{name:compare(values,all_runs['fifo']) for name,values in all_runs.items()})
print('Smoke complete: 4 policies × 2 repeats × 350 occurrences; runs/smoke/comparison.json')
