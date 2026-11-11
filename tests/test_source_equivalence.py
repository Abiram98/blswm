"""Optional actual-source differential checks; installed core needs no source mount."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from sparql_wm_bench.workloads import generate

def load(name,path):
    if not Path(path).exists(): pytest.skip('Read-only source unavailable for differential audit')
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;exec(compile(Path(path).read_text(),path,'exec'),module.__dict__);return module

def test_actual_arrival_source():
    source=load('source_arrivals','/DWNet/prof/arrival_time.py')
    bag=[dict(query_id=str(i),query_text=str(i),profile='M') for i in range(350)]
    # Consume the same random shuffle before group generation.
    rng=np.random.default_rng(2); rng.shuffle(list(range(350)))
    expected=np.cumsum(rng.exponential(1/.6,350))
    assert np.allclose([r['arrival_s'] for r in generate(bag,'P',2)],expected)
    a=source.burst_offsets(10,.5,np.random.default_rng(3))
    assert a[0]==0 and a[-1]==.5 and np.all(np.diff(a)>0)

def test_iconq_actual_source():
    source=load('source_iconq','/DWNet/iconqshed/policy.py')
    from sparql_wm_bench import iconq
    class Predictor:
        def predict_set(self,c,r,t): return {x.query_id:float(len(x.query_id)) for x in [*r,c]}
    source_wait=[source.WaitingQuery('long','q',0),source.WaitingQuery('x','q',0)]
    target_wait=[iconq.WaitingQuery('long','q',0),iconq.WaitingQuery('x','q',0)]
    cfg=dict(max_concurrency_level=10)
    for running in (False,True):
        sr=[source.RunningQuery('run','q',0,10,10)] if running else []
        tr=[iconq.RunningQuery('run','q',0,10,10)] if running else []
        a=source.IconQSchedPolicy(Predictor(),source.PolicyConfig(**cfg)).select(source_wait,sr,1)
        b=iconq.IconQSchedPolicy(Predictor(),iconq.PolicyConfig(**cfg)).select(target_wait,tr,1)
        assert a.query_id==b.query_id and a.reason==b.reason

def test_actual_capacity_and_rr_source():
    import ast
    from collections import namedtuple
    from sparql_wm_bench.policies import Policy
    from sparql_wm_bench.schema import Occurrence,Running,FakePredictor
    path=Path('/DWNet/wrk_man/policy/scheduling/policies.py')
    if not path.exists(): pytest.skip('Source unavailable')
    tree=ast.parse(path.read_text())
    keep={'RTQScheduler','SQAScheduler','FIFOScheduler'}
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in keep],type_ignores=[])
    Admission=namedtuple('Admission','resource prediction cost',defaults=(None,None))
    ns=dict(Scheduler=object,Admission=Admission)
    exec(compile(module,str(path),'exec'),ns)
    source=ns['RTQScheduler']([SimpleNamespace(mpl=5),SimpleNamespace(mpl=5)])
    candidate=Occurrence('c','c',0,0)
    for fast in range(6):
        for slow in range(6):
            running=tuple(Running(Occurrence(f'{slot}_{i}','r',0,i),0,slot=slot) for slot,n in enumerate((fast,slow)) for i in range(n))
            for cls in (0,1):
                expected=source.admit(cls,[fast,slow])
                actual=Policy('rtq').decide((candidate,),running,0,FakePredictor(default=cls),{})
                assert (actual.slot if actual else None)==(expected.resource if expected else None)
            expected=ns['SQAScheduler'](10,max_workers=10).admit(3,[fast,slow],used_cost=fast+slow,cost=3)
            actual=Policy('sqa').decide((candidate,),running,0,FakePredictor(default=1),{})
            assert bool(expected)==bool(actual)
