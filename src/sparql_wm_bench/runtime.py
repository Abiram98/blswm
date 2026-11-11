"""Open-loop replay with a single controller and injected executors/clocks."""
import heapq
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from .io import digest
from .schema import Occurrence, Running, validate_predictions

class HTTPExecutor:
    def __init__(self, endpoint, timeout=300, headers=None):
        self.endpoint,self.timeout,self.headers=endpoint,timeout,headers or {}
    def execute(self,text):
        req=Request(self.endpoint,data=urlencode({'query':text}).encode(),headers={'Accept':'application/sparql-results+json',**self.headers})
        with urlopen(req,timeout=self.timeout) as response:
            while response.read(65536): pass

class MeasuredPredictor:
    def __init__(self, predictor, clock, logs, task='classification'):
        self.predictor,self.clock,self.logs,self.task=predictor,clock,logs,task
    def _call(self, method, candidate, running, now):
        start=self.clock()
        keys=[r.occurrence.occurrence_id for r in running]+[candidate.occurrence_id]
        result=getattr(self.predictor,method)(candidate,running,now)
        values=result if method=='predict_joint' else {candidate.occurrence_id:result}
        validate_predictions(values,keys if method=='predict_joint' else [candidate.occurrence_id],self.task)
        raw_values=dict(values)
        if self.task=='regression':
            values={k:max(1e-6,float(v)) for k,v in values.items()}
            result=values if method=='predict_joint' else values[candidate.occurrence_id]
        self.logs.append(dict(method=method,occurrence_id=candidate.occurrence_id,state_keys=keys,
                              start_s=start,duration_s=self.clock()-start,predictions=values,raw_predictions=raw_values,regression_floor_s=1e-6 if self.task=='regression' else None))
        return result
    def predict(self,*args): return self._call('predict',*args)
    def predict_joint(self,*args): return self._call('predict_joint',*args)

def replay(trace, catalog, policy, predictor, *, executor=None, durations=None, inference_s=0, run_id='run1'):
    """durations explicitly selects synthetic simulation; never used in real mode."""
    occurrences=[Occurrence(str(r['occurrence_id']),str(r['query_id']),float(r['arrival_s']),int(r['trace_index'])) for r in trace]
    if len({o.trace_index for o in occurrences})!=len(occurrences): raise ValueError('Duplicate trace index')
    if len({o.occurrence_id for o in occurrences})!=len(occurrences): raise ValueError('Duplicate occurrence')
    if any(not math.isfinite(o.arrival_s) or o.arrival_s<0 for o in occurrences): raise ValueError('Invalid arrival')
    occurrences.sort(key=lambda o:(o.arrival_s,o.trace_index))
    queries={r['query_id']:r['query_text'] for r in catalog}
    metadata={k:catalog[0].get(k) if catalog else None for k in ('dataset','workload','scenario')}
    if any(o.query_id not in queries for o in occurrences): raise ValueError('Missing query text')
    simulated=durations is not None
    if not simulated and executor is None: raise ValueError('Real replay requires executor')
    if inference_s<0: raise ValueError('Negative inference cost')
    origin=time.monotonic(); now=0.; cursor=0; waiting=[]; running=[]; results=[]; predictions=[]; decisions=[]; state={}; pending=[]
    clock=lambda: now if simulated else time.monotonic()-origin
    measured=MeasuredPredictor(predictor,clock,predictions,'regression' if policy.name=='iconq' else 'classification')
    pool=ThreadPoolExecutor(max_workers=policy.mpl) if not simulated else None
    def work(text):
        try: executor.execute(text); status='success'; error=None
        except Exception as e: status='failed'; error=f'{type(e).__name__}: {e}'
        return clock(),status,error
    def complete(r,end,status='success',error=None):
        o=r.occurrence
        results.append(dict(run_id=run_id,policy=policy.name,occurrence_id=o.occurrence_id,query_id=o.query_id,
                            scheduled_arrival=o.arrival_s,arrival_s=o.arrival_s,admission_s=r.admission_s,completion_s=end,
                            wait_s=r.admission_s-o.arrival_s,execution_s=end-r.admission_s,latency_s=end-o.arrival_s,
                            status=status,error=error,dataset=metadata.get('dataset'),workload=metadata.get('workload'),scenario=metadata.get('scenario'),slot=r.slot,cost=r.cost,prediction=r.prediction))
        running.remove(r)
    controller_error=None
    try:
        while cursor<len(occurrences) or waiting or running:
            if not simulated: now=clock()
            if simulated:
                while pending and pending[0][0]<=now:
                    end,_,r=heapq.heappop(pending); complete(r,end)
            else:
                for future,r in list(pending):
                    if future.done():
                        end,status,error=future.result(); complete(r,end,status,error); pending.remove((future,r))
            while cursor<len(occurrences) and occurrences[cursor].arrival_s<=now:
                waiting.append(occurrences[cursor]); cursor+=1
            if cursor==len(occurrences) and not waiting and not running: break
            started=clock(); previous=len(predictions)
            proposed_state=copy.deepcopy(state)
            d=policy.decide(tuple(waiting),tuple(running),now,measured,proposed_state)
            if simulated:
                for index,log in enumerate(predictions[previous:]):
                    log['start_s']=started+index*inference_s
                    log['duration_s']=inference_s
                    log['timing_mode']='simulated'
                now+=inference_s*(len(predictions)-previous)
            else: now=clock()
            decisions.append(dict(time_s=now,duration_s=now-started,waiting=len(waiting),running=len(running),
                                  admitted=d.occurrence.occurrence_id if d else None))
            # Processing can span arrivals/completions; reconcile before committing a decision.
            stale=any((entry[0]<=now if simulated else entry[0].done()) for entry in pending)
            if stale: continue
            if d:
                state=proposed_state
                waiting.remove(d.occurrence)
                r=Running(d.occurrence,now,d.slot,d.cost,d.prediction); running.append(r)
                if simulated:
                    duration=float(durations[d.occurrence.query_id])
                    if duration<0 or not math.isfinite(duration): raise ValueError('Invalid synthetic duration')
                    heapq.heappush(pending,(now+duration,d.occurrence.trace_index,r))
                else: pending.append((pool.submit(work,queries[d.occurrence.query_id]),r))
                continue
            if simulated:
                wake=[occurrences[cursor].arrival_s] if cursor<len(occurrences) else []
                if pending: wake.append(pending[0][0])
                if not wake: raise RuntimeError('Policy deadlock: waiting without future event')
                now=max(now,min(wake))
            else:
                if waiting and not running and cursor==len(occurrences): raise RuntimeError('Policy deadlock')
                delay=min(.01,max(0,occurrences[cursor].arrival_s-now)) if cursor<len(occurrences) else .001
                time.sleep(delay)
    except Exception as exc:
        controller_error=f'{type(exc).__name__}: {exc}'
        if simulated:
            while pending:
                end,_,r=heapq.heappop(pending); complete(r,end)
        else:
            for future,r in pending:
                end,status,error=future.result(); complete(r,end,status,error)
        recorded={r['occurrence_id'] for r in results}
        for o in occurrences:
            if o.occurrence_id not in recorded:
                results.append(dict(run_id=run_id,policy=policy.name,occurrence_id=o.occurrence_id,query_id=o.query_id,
                                    scheduled_arrival=o.arrival_s,arrival_s=o.arrival_s,admission_s=None,completion_s=None,
                                    wait_s=None,execution_s=None,latency_s=None,status='controller_failure',error=controller_error,
                                    dataset=metadata.get('dataset'),workload=metadata.get('workload'),scenario=metadata.get('scenario'),slot=None,cost=None,prediction=None))
    finally:
        if pool: pool.shutdown(wait=True)
    return dict(records=results,predictions=predictions,decisions=decisions,trace=trace,run_id=run_id,
                controller_error=controller_error,setting_id=f'{policy.name}_mpl{policy.mpl}',mpl=policy.mpl,catalog_hash=digest(queries),
                mode='synthetic' if simulated else 'real',complete=controller_error is None and len(results)==len(occurrences) and all(r['status']=='success' for r in results))
