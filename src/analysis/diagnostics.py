"""Run-level queue and admission diagnostics from recorded observations."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from wrk_gen.queries import fingerprint


def run_diagnostics(frame, path, meta, *, complete):
    result=dict(backlogged_duration_s=None,mean_running_while_backlogged=None,prediction_calls=None,
                median_prediction_call_s=None,admission_regression_count=None,admission_q_error_count=None,
                admission_median_q_error=None,admission_median_prediction_over_runtime=None)
    if complete and {'arrival_s','start_s','end_s'} <= set(frame.columns):
        times=frame[['arrival_s','start_s','end_s']].apply(pd.to_numeric,errors='coerce')
        if times.notna().all().all() and np.isfinite(times.to_numpy()).all() and (times.arrival_s<=times.start_s).all() and (times.start_s<=times.end_s).all():
            events={}
            for arrival,start,end in times.itertuples(index=False,name=None):
                for t,dw,dr in ((arrival,1,0),(start,-1,1),(end,0,-1)):
                    previous=events.get(t,(0,0));events[t]=(previous[0]+dw,previous[1]+dr)
            waiting=running=0;backlog=area=0.;last=None
            for t,(dw,dr) in sorted(events.items()):
                if last is not None and waiting>0:
                    interval=t-last;backlog+=interval;area+=interval*running
                waiting+=dw;running+=dr;last=t
            result.update(backlogged_duration_s=backlog,mean_running_while_backlogged=area/backlog if backlog>0 else None)
    source=Path(path)/'run.json'
    if source.is_file():
        if meta.get('input_sha256',{}).get(str(source))!=fingerprint(source):raise ValueError(f'Run changed since preparation: {source}')
        payload=json.loads(source.read_text());calls=payload.get('predictions',[])
        durations=[x['duration_s'] for x in calls if isinstance(x.get('duration_s'),(int,float)) and math.isfinite(x['duration_s']) and x['duration_s']>=0]
        result.update(prediction_calls=len(calls),median_prediction_call_s=float(np.median(durations)) if durations else None)
        if calls and any(x.get('regression_floor_s') is not None for x in calls):
            pairs=[(r.get('admission_prediction'),r.get('execution_s')) for r in payload.get('records',[]) if r.get('status')=='success']
            pairs=[(p,y) for p,y in pairs if isinstance(p,(int,float)) and isinstance(y,(int,float)) and math.isfinite(p) and math.isfinite(y)]
            positive=[(p,y) for p,y in pairs if p>0 and y>0]
            result.update(admission_regression_count=len(pairs),admission_q_error_count=len(positive),
                admission_median_q_error=float(np.median([max(p/y,y/p) for p,y in positive])) if positive else None,
                admission_median_prediction_over_runtime=float(np.median([p/y for p,y in positive])) if positive else None)
    return result
