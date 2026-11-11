"""One JSONL state schema shared by training and inference adapters."""
import random
import math
from .catalog import identity

def validate(sample):
    s=sample['S']; y=sample['y']
    if len(s)!=len(y) or len({x['occurrence_id'] for x in s})!=len(s):
        raise ValueError('Invalid aligned state')
    if any(not math.isfinite(float(v)) or float(v)<0 for v in y): raise ValueError('Negative execution label')
    return sample

def build(run, catalog, workload_id='W1', mpl=10):
    if run.get('policy','fifo')!='fifo' and any(r['policy']!='fifo' for r in run['records']):
        raise ValueError('CQPP requires FIFO traces')
    texts={r['query_id']:r['query_text'] for r in catalog}
    records=run['records']
    if any(r['status']!='success' for r in records): raise ValueError('Incomplete training run')
    events=[]
    for i,r in enumerate(records):
        events.extend([(r['completion_s'],0,i,r),(r['admission_s'],1,i,r)])
    active=[]; samples=[]
    for t,kind,i,r in sorted(events,key=lambda e:e[:3]):
        if kind==0:
            active=[x for x in active if x['occurrence_id']!=r['occurrence_id']]
            state=active[:]
        else:
            state=active+[r]; active.append(r)
        if not state: continue
        samples.append(validate(dict(sample_id=f"{r['run_id']}:{len(samples)}",run_id=r['run_id'],workload_id=workload_id,mpl=mpl,
                           event='admission' if kind else 'completion',decision_s=t,
                           S=[dict(occurrence_id=x['occurrence_id'],query_id=x['query_id'],query_text=texts[x['query_id']],
                                   admission_s=x['admission_s'],role='candidate' if kind and x is r else 'running') for x in state],
                           y=[x['completion_s']-x['admission_s'] for x in state])))
    return samples

def split(samples, test=(), rho=.2, seed=1):
    if not 0<rho<1: raise ValueError('Invalid validation fraction')
    train_ids={identity(x['query_text']) for s in samples for x in s['S']}
    test_ids={identity(x['query_text']) for s in test for x in s['S']}
    if train_ids & test_ids: raise ValueError('Training/test query-text leakage')
    # Group entire run; overlapping states never cross the validation boundary.
    groups=sorted({s['run_id'] for s in samples})
    if len(groups)<2: raise ValueError('At least two trace runs needed for grouped split')
    random.Random(seed).shuffle(groups)
    val=set(groups[:max(1,min(len(groups)-1,round(len(groups)*rho)))])
    return [s for s in samples if s['run_id'] not in val],[s for s in samples if s['run_id'] in val],dict(seed=seed,rho=rho,validation_runs=sorted(val),test_query_overlap=0)
