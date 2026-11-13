"""Analyze recorded or TCP predictions against observed execution labels."""
import json
import ast
import csv
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
from wrk_gen.queries import fingerprint
from .io import output_directory
from .tables import write_latex


def analyze(data,output,*,report=None,host=None,port=None,timeout=60.,thresholds=None):
    source=Path(data).resolve(); hashes={str(source):fingerprint(source)}
    states={};candidate_data=source.suffix.lower() in ('.csv','.tsv')
    if candidate_data:
        csv.field_size_limit(sys.maxsize)
        with source.open(newline='',encoding='utf-8-sig') as f:
            for index,row in enumerate(csv.DictReader(f,delimiter='\t' if source.suffix.lower()=='.tsv' else ','),1):
                text=row.get('query_text') or row.get('queryString')
                try:
                    runtime=float(row.get('Execution_Time') or row.get('mean_latency'))
                    running=ast.literal_eval(row.get('Running_Queries_At_Start') or '[]')
                except (ValueError,TypeError,SyntaxError) as exc:raise ValueError(f'Invalid candidate CSV row {index}') from exc
                if not text or not math.isfinite(runtime) or runtime<0 or not isinstance(running,list):raise ValueError(f'Invalid candidate CSV row {index}')
                queries=[]
                for entry in running:
                    if isinstance(entry,str):entry={'text':entry}
                    if not isinstance(entry,dict):raise ValueError('Invalid running query')
                    query_text=entry.get('query_text') or entry.get('text')
                    if not query_text:raise ValueError('Running query has no text')
                    queries.append(dict(query_id=str(entry.get('query_id','running')),query_text=query_text,role='running',pos=len(queries),elapsed_at_decision=0.))
                queries.append(dict(query_id=str(row.get('Query_ID') or row.get('id') or index),query_text=text,role='candidate',pos=len(queries),elapsed_at_decision=0.,observed_runtime=runtime))
                states[str(index)]=dict(sample_id=str(index),S=queries,y=[None]*(len(queries)-1)+[None],set_size=len(queries))
    else:
        with source.open() as f:
            for line in f:
                if not line.strip():continue
                sample=json.loads(line);key=str(sample['sample_id'])
                if key in states:raise ValueError(f'Duplicate sample_id: {key}')
                if len(sample['S'])!=len(sample['y']) or len(sample['S'])!=sample.get('set_size',len(sample['S'])):
                    raise ValueError(f'Invalid aligned sample: {key}')
                states[key]=sample
    if not states:raise ValueError('Empty dataset')
    if report:
        source_report=Path(report).resolve();hashes[str(source_report)]=fingerprint(source_report)
        payload=json.loads(source_report.read_text());entries=payload.get('predictions')
        if not isinstance(entries,list):raise ValueError('Evaluation report has no predictions list')
        task=payload.get('prediction_task','regression' if 'rmse' in payload else 'classification')
        used_thresholds=thresholds
        expects_all=payload.get('model_type') not in ('planrgcn','dwns')
    else:
        if host is None or port is None:raise ValueError('Specify --report or --host and --port')
        from qpp.protocol import request
        info=request(host,port,{'type':'info'},timeout);task=info['prediction_task']
        used_thresholds=thresholds if thresholds is not None else info.get('runtime_thresholds')
        expects_all=info.get('returns_per_query_predictions') is True
        entries=[]
        for sample in states.values():
            queries=[{key:q[key] for key in ('query_id','query_text','role','pos','elapsed_at_decision','start_time') if key in q} for q in sample['S']]
            response=request(host,port,dict(type='predict',sample_id=sample['sample_id'],queries=queries),timeout)
            entries.append(dict(sample_id=sample['sample_id'],predictions=response['predictions']))
    if task not in ('classification','regression'):raise ValueError('Unsupported prediction task')
    if used_thresholds is not None and (any(not math.isfinite(t) or t<=0 for t in used_thresholds) or used_thresholds!=sorted(set(used_thresholds))):raise ValueError('Thresholds must be positive, sorted and unique')
    if candidate_data and task=='classification' and used_thresholds is None:
        raise ValueError('Candidate CSV classification requires --thresholds when analyzing a report')
    rows=[];seen=set();unlabeled=0;returned=0
    for entry in entries:
        key=str(entry['sample_id'])
        if key not in states or key in seen:raise ValueError(f'Unknown or duplicate prediction sample: {key}')
        seen.add(key);sample=states[key]
        predictions=entry.get('predictions')
        if predictions is None:
            predictions=[]
            for i,index in enumerate(entry['input_indices']):
                item=dict(input_index=index,query_id=entry['query_ids'][i])
                if task=='classification':item['predicted_class']=entry['predicted_classes'][i]
                else:item['predicted_runtime_seconds']=entry['predicted_runtime_seconds'][i]
                predictions.append(item)
        if not isinstance(predictions,list):raise ValueError(f'Invalid predictions: {key}')
        returned+=len(predictions)
        indices=set()
        for prediction in predictions:
            index=prediction['input_index']
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(sample['S']) or index in indices:raise ValueError(f'Invalid prediction index: {key}')
            indices.add(index);query=sample['S'][index]
            if str(prediction['query_id'])!=str(query['query_id']):raise ValueError(f'Prediction identity mismatch: {key}')
            runtime=query.get('observed_runtime')
            if candidate_data and query.get('role')=='running':
                unlabeled+=1
                continue
            if runtime is not None:
                runtime=float(runtime)
                if not math.isfinite(runtime) or runtime<0:raise ValueError('Invalid observed runtime')
            role=query.get('role','unknown')
            if role not in ('candidate','running','unknown'):raise ValueError('Invalid query role')
            base=dict(sample_id=key,query_id=str(query['query_id']),input_index=index,role=role,state_size=len(sample['S']),observed_runtime=runtime)
            if task=='classification':
                pred=prediction['predicted_class']
                if isinstance(pred,bool) or not isinstance(pred,int) or pred<0:raise ValueError('Invalid predicted class')
                truth=sum(runtime>=t for t in used_thresholds) if used_thresholds is not None and runtime is not None else sample['y'][index]
                if isinstance(truth,bool) or not isinstance(truth,int) or truth<0:raise ValueError('Invalid true class')
                probs=prediction.get('probabilities')
                if probs is not None:
                    if not isinstance(probs,dict) or any(not str(k).isdigit() or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for k,v in probs.items()) or not math.isclose(sum(probs.values()),1.,abs_tol=1e-5):
                        raise ValueError('Invalid class probabilities')
                    if str(truth) not in probs:raise ValueError('True class missing from predicted probabilities')
                base.update(true_class=truth,pred_class=pred,confidence=prediction.get('confidence'),probabilities=json.dumps(probs) if probs is not None else None)
            else:
                pred=prediction['predicted_runtime_seconds']
                if isinstance(pred,bool) or not isinstance(pred,(int,float)) or not math.isfinite(pred) or runtime is None:raise ValueError('Regression requires finite predictions and observed runtimes')
                base.update(predicted_runtime=float(pred),error=float(pred)-runtime,absolute_error=abs(float(pred)-runtime))
            rows.append(base)
        if expects_all and len(indices)!=len(sample['S']):raise ValueError(f'Joint predictor returned an incomplete state: {key}')
    if seen!=set(states):raise ValueError(f'Missing predictions for {len(set(states)-seen)} samples')
    if not rows:raise ValueError('No prediction rows')
    frame=pd.DataFrame(rows)
    labels=list(range(len(used_thresholds)+1)) if used_thresholds is not None else None
    metrics=_metrics(frame,task,labels)
    strata=[]
    for field in ('role','state_size'):
        for value,part in frame.groupby(field,dropna=False):
            strata.append(dict(group=field,value=int(value) if field=='state_size' else str(value),**_metrics(part,task,labels)))
    if any(fingerprint(path)!=digest for path,digest in hashes.items()):raise ValueError('Prediction input changed')
    destination=output_directory(output,[source,*([source_report] if report else [])])
    frame.to_csv(destination/'predictions.csv',index=False)
    (destination/'metrics.json').write_text(json.dumps(dict(task=task,thresholds=used_thresholds,overall=metrics,strata=strata,prediction_coverage=dict(returned=returned,possible=sum(len(s['S']) for s in states.values()),analyzed=len(rows)),unlabeled_predictions_excluded=unlabeled,input_sha256=hashes,source='report' if report else dict(host=host,port=port)),indent=2,allow_nan=False)+'\n')
    _plots(frame,task,destination,labels)
    for role,part in frame.groupby('role',sort=False):
        role_path=destination/str(role);role_path.mkdir()
        _plots(part,task,role_path,labels)
    if task=='classification':
        table=[]
        for role,part in frame.groupby('role',sort=False):
            result=_metrics(part,task,labels)
            table.append(dict(role=role,count=result['count'],accuracy=result['accuracy'],macro_f1=result['macro_f1'],**{f"recall_class_{c['label']}":c['recall'] for c in result['per_class']}))
        paper=pd.DataFrame(table);paper.to_csv(destination/'paper_metrics.csv',index=False)
        headers=list(paper.columns);write_latex(destination/'paper_metrics.tex',headers,paper.itertuples(index=False,name=None))
    else:
        paper=pd.DataFrame([dict(role=role,**_metrics(part,task)) for role,part in frame.groupby('role',sort=False)]);paper.to_csv(destination/'paper_metrics.csv',index=False)
        headers=['role','count','mae','rmse','median_q_error','overprediction_fraction','underprediction_fraction']
        write_latex(destination/'paper_metrics.tex',headers,paper[headers].itertuples(index=False,name=None))
        if used_thresholds:
            snapped=frame.copy();snapped['true_class']=[sum(x>=t for t in used_thresholds) for x in frame.observed_runtime];snapped['pred_class']=[sum(x>=t for t in used_thresholds) for x in frame.predicted_runtime]
            _plots(snapped,'classification',destination,labels,filename='snapped_confusion_matrix')
            for role,part in snapped.groupby('role',sort=False):
                _plots(part,'classification',destination/str(role),labels,filename='snapped_confusion_matrix')
    return dict(output=str(destination),task=task,states=len(seen),predictions=len(frame),returned_predictions=returned,possible_predictions=sum(len(s['S']) for s in states.values()),unlabeled_predictions_excluded=unlabeled,metrics=metrics)


def _metrics(frame,task,labels=None):
    if task=='regression':
        y=frame.observed_runtime.to_numpy(float);p=frame.predicted_runtime.to_numpy(float);e=p-y
        valid=(p>0)&(y>0)
        q=np.maximum(p[valid]/y[valid],y[valid]/p[valid])
        total=float(np.sum((y-y.mean())**2))
        return dict(count=len(y),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))),
                    median_absolute_error=float(np.median(abs(e))),r2=float(1-np.sum(e**2)/total) if total>0 else None,
                    median_q_error=float(np.median(q)) if len(q) else None,q_error_excluded=int((~valid).sum()),
                    nonpositive_predictions=int((p<=0).sum()),overprediction_fraction=float(np.mean(e>0)),underprediction_fraction=float(np.mean(e<0)),median_prediction_over_runtime=float(np.median(p[y>0]/y[y>0])) if (y>0).any() else None)
    y=frame.true_class.to_numpy(int);p=frame.pred_class.to_numpy(int)
    labels=sorted(set(labels or [])|{int(v) for v in set(y)|set(p)});matrix=[[int(np.sum((y==a)&(p==b))) for b in labels] for a in labels]
    f1=[];per_class=[]
    for i,label in enumerate(labels):
        support=sum(matrix[i]);predicted=sum(row[i] for row in matrix);tp=matrix[i][i]
        precision=tp/predicted if predicted else 0.;recall=tp/support if support else 0.
        score=2*tp/(support+predicted) if support+predicted else 0.;f1.append(score)
        per_class.append(dict(label=label,support=support,precision=precision,recall=recall,f1=score))
    normalized=[[v/sum(row) if sum(row) else None for v in row] for row in matrix]
    supported=[c['recall'] for c in per_class if c['support']]
    result=dict(count=len(y),accuracy=float(np.mean(y==p)),macro_f1=float(np.mean(f1)),balanced_accuracy=float(np.mean(supported)) if supported else None,labels=labels,confusion_matrix=matrix,row_normalized_confusion_matrix=normalized,per_class=per_class)
    available=frame[frame.probabilities.notna()] if 'probabilities' in frame else frame.iloc[:0]
    if len(available):
        logloss=[];brier=[];calibration=[]
        for row in available.itertuples():
            probabilities=json.loads(row.probabilities);truth=str(row.true_class)
            logloss.append(-math.log(max(float(probabilities[truth]),1e-15)))
            brier.append(sum((float(value)-(key==truth))**2 for key,value in probabilities.items()))
            calibration.append((max(probabilities.values()),row.true_class==row.pred_class))
        ece=0.
        for index in range(10):
            group=[(conf,correct) for conf,correct in calibration if (index/10<=conf<(index+1)/10 or index==9 and conf==1.)]
            if group:ece+=len(group)/len(calibration)*abs(sum(c for c,_ in group)/len(group)-sum(ok for _,ok in group)/len(group))
        result.update(probability_rows=len(available),cross_entropy=float(np.mean(logloss)),brier_score=float(np.mean(brier)),expected_calibration_error=float(ece))
    return result


def _plots(frame,task,destination,labels=None,filename=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    if task=='classification':
        metrics=_metrics(frame,task,labels);labels=metrics['labels']
        matrix=np.array([[np.nan if v is None else v for v in row] for row in metrics['row_normalized_confusion_matrix']]);im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1)
        names=['Fast' if v==0 and len(labels)==2 else 'Slow' if v==1 and len(labels)==2 else str(v) for v in labels]
        ax.set_xticks(range(len(labels)),names);ax.set_yticks(range(len(labels)),names)
        for i,row in enumerate(metrics['confusion_matrix']):
            for j,count in enumerate(row):
                val=matrix[i,j];text=f'{val:.2f}\n(n={count})' if np.isfinite(val) else f'N/A\n(n={count})'
                ax.text(j,i,text,ha='center',va='center',color='white' if val>=.55 else 'black')
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class');fig.colorbar(im,ax=ax,label='Row-normalized rate');name=filename or 'confusion_matrix'
        pd.DataFrame([dict(true_class=labels[i],predicted_class=labels[j],count=count,row_normalized=metrics['row_normalized_confusion_matrix'][i][j]) for i,row in enumerate(metrics['confusion_matrix']) for j,count in enumerate(row)]).to_csv(destination/f'{name}.csv',index=False)
    else:
        ax.scatter(frame.observed_runtime,frame.predicted_runtime,alpha=.65)
        limits=[min(frame.observed_runtime.min(),frame.predicted_runtime.min()),max(frame.observed_runtime.max(),frame.predicted_runtime.max())]
        ax.plot(limits,limits,'--',color='black');ax.set_xlabel('Observed runtime (s)');ax.set_ylabel('Predicted runtime (s)');name='predicted_vs_observed'
    fig.tight_layout()
    fig.savefig(destination/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)
    if task=='regression':
        fig,ax=plt.subplots(figsize=(6,4));x=np.sort(frame.absolute_error.to_numpy(float))
        ax.step(x,np.arange(1,len(x)+1)/len(x),where='post');ax.set_xlabel('Absolute error (s)');ax.set_ylabel('Cumulative fraction')
        fig.tight_layout();fig.savefig(destination/'absolute_error_cdf.pdf',bbox_inches='tight');plt.close(fig)
