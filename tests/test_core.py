import pytest
from wrk_gen.profiling import classify as profile
from policies.schema import FakePredictor, Occurrence, Running
from policies.scheduling import Policy

@pytest.mark.parametrize('t,c,p',[(0,None,'S'),(1,100,'M'),(9,200,'M'),(1,201,'HM'),(10,200,'L'),(100,200,'L100'),(110,200,'L100'),(110.001,200,'L'),(120,200,'L'),(120,201,'HL'),(120.1,200,None),(1,None,None),(1,99,None),(-1,200,None),(1,float('nan'),None)])
def test_profiles(t,c,p): assert profile(t,c)==p






def test_rtq_bypass_and_physical():
    slow=Occurrence('slow','s',0,0); fast=Occurrence('fast','f',0,1)
    active=tuple(Running(Occurrence(str(i),str(i),0,i),0,slot=1) for i in range(5))
    pred=FakePredictor({'s':1,'f':0})
    assert Policy('rtq').decide((slow,fast),active,0,pred,{}).occurrence==fast
    assert Policy('rtq-reeval').decide((slow,),active,0,pred,{}) is None







def test_reeval_budget_and_joint_order():
    a=Occurrence('a','a',0,0); b=Occurrence('b','b',0,1)
    running=tuple(Running(Occurrence(f'r{i}',f'r{i}',0,i+2),0,cost=1) for i in range(3))
    pred=FakePredictor(default=1)
    assert Policy('sqa').decide((a,),running,0,pred,{}) is not None
    assert Policy('sqa-reeval').decide((a,),running,0,pred,{}) is None
