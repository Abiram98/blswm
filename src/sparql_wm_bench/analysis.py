"""Offline metrics: imports neither execution nor ML code."""
from collections import defaultdict
import numpy as np

def summarize(records):
    success=[r for r in records if r['status']=='success']
    out=dict(offered=len(records),completed=len(success),failures=len(records)-len(success))
    for field in ('wait_s','execution_s','latency_s'):
        values=[float(r[field]) for r in success]
        out[field]=dict(zip(('p10','p25','p50','p75','p95'),map(float,np.percentile(values,[10,25,50,75,95])))) if values else None
        if values: out[field]['mean']=float(np.mean(values))
    span=max((r['completion_s'] for r in records if r['completion_s'] is not None),default=0)-min((r['arrival_s'] for r in records),default=0)
    out['throughput']=len(success)/span if span>0 else None
    return out

def averaged(runs):
    if not runs: raise ValueError('No runs')
    keys={r['occurrence_id'] for r in runs[0]['records']}
    trace=runs[0]['trace']
    for run in runs:
        if not run['complete'] or run['trace']!=trace or {r['occurrence_id'] for r in run['records']}!=keys:
            raise ValueError('Incomplete or unmatched repeat; exclusions are not permitted')
    return {k:float(np.mean([next(r['latency_s'] for r in run['records'] if r['occurrence_id']==k) for run in runs])) for k in sorted(keys)}

def compare(policy, reference):
    if policy.keys()!=reference.keys(): raise ValueError('Unmatched occurrences; exclusions are not permitted')
    if any(v<=0 for v in reference.values()): raise ValueError('Nonpositive reference latency')
    ratios=[policy[k]/reference[k] for k in policy]
    grid=sorted(set(policy.values())|set(reference.values()))
    p=np.array(list(policy.values())); r=np.array(list(reference.values()))
    return dict(S50=float(np.median(ratios)),ratios=ratios,
                delta_cdf=[dict(t=t,delta=float(np.mean(p<=t)-np.mean(r<=t))) for t in grid])

def prediction_metrics(labels, values, task='classification'):
    y=np.asarray(labels); p=np.asarray(values)
    if not len(y): raise ValueError('Empty predictions')
    if task=='regression':
        return dict(MAE=float(np.mean(abs(y-p))),RMSE=float(np.sqrt(np.mean((y-p)**2))))
    confusion=[[int(np.sum((y==i)&(p==j))) for j in (0,1)] for i in (0,1)]
    recall=[]; f1=[]
    for i in (0,1):
        tp=confusion[i][i]; support=sum(confusion[i]); predicted=sum(row[i] for row in confusion)
        recall.append(tp/support if support else 0.)
        f1.append(2*tp/(support+predicted) if support+predicted else 0.)
    return dict(confusion=confusion,recall=recall,macro_F1=float(np.mean(f1)),support=[sum(row) for row in confusion])

def plot(groups, path):
    from .io import output
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].boxplot([list(v.values()) for v in groups.values()],labels=list(groups),whis=1.5)
    for name,v in groups.items():
        x=np.sort(list(v.values())); axes[1].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=name)
    axes[0].set_ylabel('Mean latency per occurrence (s)'); axes[1].legend(); axes[1].set_xlabel('Latency (s)')
    fig.tight_layout(); fig.savefig(output(path)); plt.close(fig)

def diagnostics(run):
    records=[r for r in run['records'] if r['admission_s'] is not None and r['completion_s'] is not None]; points=sorted({r[k] for r in records for k in ('arrival_s','admission_s','completion_s')})
    occupied=[0.,0.]; backlogged=0.; integral=0.
    for left,right in zip(points,points[1:]):
        active=[r for r in records if r['admission_s']<=left<r['completion_s']]
        dt=right-left
        for slot in (0,1): occupied[slot]+=dt*sum(r['slot']==slot for r in active)
        if any(r['arrival_s']<=left<r['admission_s'] for r in records):
            backlogged+=dt; integral+=dt*len(active)
    return dict(time_weighted_running_while_backlogged=integral/backlogged if backlogged else None,
                slot_seconds=occupied,prediction_calls=len(run.get('predictions',[])),
                inference_s=sum(p['duration_s'] for p in run.get('predictions',[])),
                scheduler_s=sum(d['duration_s'] for d in run.get('decisions',[])),
                unavailable=['correctness of result sets requires external oracle'])
