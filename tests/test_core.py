import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest
from sparql_wm_bench.catalog import profile
from sparql_wm_bench.workloads import COMPOSITIONS,PROFILES,generate,select
from sparql_wm_bench.schema import FakePredictor,Occurrence,Running
from sparql_wm_bench.policies import Policy
from sparql_wm_bench.runtime import replay
from sparql_wm_bench.analysis import compare,averaged
from sparql_wm_bench.dataset import build,split
from sparql_wm_bench.io import output

@pytest.mark.parametrize('t,c,p',[(0,None,'S'),(1,100,'M'),(9,200,'M'),(1,201,'HM'),(10,200,'L'),(100,200,'L100'),(110,200,'L100'),(110.001,200,'L'),(120,200,'L'),(120,201,'HL'),(120.1,200,None),(1,None,None),(1,99,None),(-1,200,None),(1,float('nan'),None)])
def test_profiles(t,c,p): assert profile(t,c)==p

def bag(comp):
    return [dict(query_id=f'{p}{i}',query_text=f'SELECT * WHERE {{ ?s <urn:{p}{i}> ?o }}',profile=p,isolated_runtime_s=2) for p,n in zip(PROFILES,COMPOSITIONS[comp]) for i in range(n)]

@pytest.mark.parametrize('comp',COMPOSITIONS)
def test_patterns(comp):
    b=bag(comp)
    for pattern in ['P','PAR']+(['LF'] if comp!='Homogeneous' else [])+(['RCB'] if comp in ['CPU-demanding','Heterogeneous'] else []):
        t=generate(b,pattern,3)
        assert t==generate(b,pattern,3)
        assert len(t)==len({r['occurrence_id'] for r in t})==350
        assert sorted(r['query_id'] for r in t)==sorted(r['query_id'] for r in b)
        assert all(t[i]['arrival_s']<t[i+1]['arrival_s'] for i in range(349))
        if pattern=='LF': assert all(r['query_id'].startswith('L100') for r in t[:10])
        if pattern=='RCB':
            for start in (106,223,340):
                assert all(r['query_id'].startswith('HM') for r in t[start:start+10])
                assert t[start+9]['arrival_s']-t[start]['arrival_s']==pytest.approx(.5)

def fixture(n=20):
    catalog=[dict(query_id=str(i),query_text=f'SELECT {i}') for i in range(n)]
    trace=[dict(occurrence_id=f'o{i}',query_id=str(i),arrival_s=i*.1,trace_index=i) for i in range(n)]
    return catalog,trace

@pytest.mark.parametrize('name',['fifo','rr','rtq','rtq-reeval','sqa','sqa-reeval','iconq'])
def test_replay(name):
    catalog,trace=fixture()
    result=replay(trace,catalog,Policy(name),FakePredictor(default=2 if name=='iconq' else 0),durations={str(i):2 for i in range(20)})
    assert result['complete']
    assert len(result['records'])==20
    for r in result['records']:
        assert r['latency_s']==pytest.approx(r['wait_s']+r['execution_s'])
    for d in result['decisions']: assert d['running']<=10

def test_inference_wait():
    catalog,trace=fixture(1)
    run=replay(trace,catalog,Policy('sqa'),FakePredictor(),durations={'0':1},inference_s=.5)
    assert run['records'][0]['wait_s']==.5

def test_rtq_bypass_and_physical():
    slow=Occurrence('slow','s',0,0); fast=Occurrence('fast','f',0,1)
    active=tuple(Running(Occurrence(str(i),str(i),0,i),0,slot=1) for i in range(5))
    pred=FakePredictor({'s':1,'f':0})
    assert Policy('rtq').decide((slow,fast),active,0,pred,{}).occurrence==fast
    assert Policy('rtq-reeval').decide((slow,),active,0,pred,{}) is None

def test_s50():
    p={'a':1,'b':100,'c':3}; r={'a':1,'b':2,'c':3}
    assert compare(p,r)['S50']==1
    assert np.median(list(p.values()))/np.median(list(r.values()))!=1
    with pytest.raises(ValueError): compare(p,{'a':1})

def test_dataset():
    cat,trace=fixture(3)
    run=replay(trace,cat,Policy('fifo',mpl=2),FakePredictor(),durations={str(i):1 for i in range(3)})
    samples=build(run,cat)
    for s in samples:
        assert all(y==pytest.approx(1) for y in s['y'])
    assert any(len(s['S'])==2 for s in samples)
    with pytest.raises(ValueError): split(samples)
    for s in samples: s['run_id']='r1'
    more=[dict(s,run_id='r2') for s in samples]
    tr,va,report=split(samples+more)
    assert {s['run_id'] for s in tr}.isdisjoint(s['run_id'] for s in va)
    with pytest.raises(ValueError): split(samples+more,samples)

def test_output_escape(tmp_path):
    with pytest.raises(ValueError): output('../escape')
    with pytest.raises(ValueError): output('/data/escape')
    link=tmp_path/'external'; link.symlink_to('/data',target_is_directory=True)
    with pytest.raises(ValueError): output(link/'escape')

def test_insufficient():
    with pytest.raises(ValueError): select(bag('Homogeneous')[:10],{'composition':'Homogeneous'})

def test_repeat_mismatch():
    with pytest.raises(ValueError): averaged([dict(complete=False,records=[],trace=[])])

def test_fifo_order_and_capacity():
    cat,trace=fixture(12)
    for r in trace: r['arrival_s']=0
    run=replay(trace,cat,Policy('fifo',mpl=2),FakePredictor(),durations={str(i):1 for i in range(12)})
    assert [r['occurrence_id'] for r in sorted(run['records'],key=lambda r:(r['admission_s'],int(r['query_id'])))]==[f'o{i}' for i in range(12)]
    assert max(d['running'] for d in run['decisions'])<=2

def test_reeval_budget_and_joint_order():
    a=Occurrence('a','a',0,0); b=Occurrence('b','b',0,1)
    running=tuple(Running(Occurrence(f'r{i}',f'r{i}',0,i+2),0,cost=1) for i in range(3))
    pred=FakePredictor(default=1)
    assert Policy('sqa').decide((a,),running,0,pred,{}) is not None
    assert Policy('sqa-reeval').decide((a,),running,0,pred,{}) is None

def test_percentile_counts_and_disjointness():
    rows=[dict(query_id=str(i),query_text=f'SELECT {i}',isolated_runtime_s=float(i),dataset='x') for i in range(5000)]
    first=select(rows,dict(selection_seed=1,dataset='x'))
    second=select(rows,dict(selection_seed=2,dataset='x'),first)
    cutoff=np.percentile([r['isolated_runtime_s'] for r in rows],85)
    assert sum(r['isolated_runtime_s']<cutoff for r in first)==320
    assert len(first)==len(second)==350
    assert {r['query_id'] for r in first}.isdisjoint(r['query_id'] for r in second)

def test_backlog_diagnostics():
    from sparql_wm_bench.analysis import diagnostics
    cat,trace=fixture(2)
    run=replay(trace,cat,Policy('fifo',mpl=1),FakePredictor(),durations={'0':1,'1':1})
    assert diagnostics(run)['time_weighted_running_while_backlogged']==pytest.approx(1)

def test_predictor_error_records_all_offered():
    class Broken(FakePredictor):
        def predict(self,*a): raise TimeoutError('prediction timeout')
    cat,trace=fixture(3)
    run=replay(trace,cat,Policy('rtq'),Broken(),durations={str(i):1 for i in range(3)})
    assert not run['complete'] and len(run['records'])==3
    assert all(r['status']=='controller_failure' for r in run['records'])
