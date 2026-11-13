"""Compare prediction analyses over the same candidate observations."""
import json
from pathlib import Path
import pandas as pd
from .io import output_directory
from .tables import write_latex


def compare(results, output):
    inputs=[Path(p).expanduser().resolve() for p in results]
    if not inputs:raise ValueError('Supply at least one prediction analysis directory')
    names=[p.name.removesuffix('_analysis') for p in inputs]
    if len(set(names))!=len(names):raise ValueError('Prediction model names must be unique')
    loaded=[];reference=None;reference_hash=None;thresholds=None
    for name,path in zip(names,inputs):
        metadata=json.loads((path/'metrics.json').read_text())
        frame=pd.read_csv(path/'predictions.csv',dtype={'sample_id':str,'query_id':str})
        candidate=frame[frame.role=='candidate']
        keys=['sample_id','query_id','input_index']
        if candidate.empty or candidate.duplicated(keys).any():raise ValueError(f'Invalid candidate observations: {path}')
        identity=set(map(tuple,candidate[keys].to_numpy()))
        data_hash=next(iter(metadata['input_sha256'].values()))
        if reference is None:reference=identity;reference_hash=data_hash;thresholds=metadata['thresholds']
        if identity!=reference or data_hash!=reference_hash or metadata['thresholds']!=thresholds:
            raise ValueError('Models must use identical candidate observations, dataset and thresholds')
        loaded.append((name,path,metadata))
    dest=output_directory(output,inputs);rows=[]
    for name,path,metadata in loaded:
        for item in metadata['strata']:
            if item['group']!='role':continue
            result=dict(model=name,task=metadata['task'],role=item['value'],count=item['count'])
            if metadata['task']=='classification':
                result.update(accuracy=item['accuracy'],macro_f1=item['macro_f1'],**{f"recall_class_{c['label']}":c['recall'] if c['support'] else None for c in item['per_class']})
            else:
                result.update(mae=item['mae'],rmse=item['rmse'],median_q_error=item['median_q_error'],overprediction_fraction=item['overprediction_fraction'],underprediction_fraction=item['underprediction_fraction'])
            rows.append(result)
    table=pd.DataFrame(rows);table.to_csv(dest/'model_metrics.csv',index=False)
    classification=table[(table.task=='classification')&(table.role=='candidate')]
    if not classification.empty:
        display={'qsps':'Bi-SCQPP','dwns':'SCQPP','p-iconq':'P+IconQ','planrgcn':'PlanRGCN'}
        columns=[display.get(name,name) for name in classification.model]; metrics=[c for c in classification.columns if c.startswith('recall_class_')]+['macro_f1']
        values=[[('N/A' if pd.isna(v) else f'{100*v:.1f}') for v in classification[metric]] for metric in metrics]
        def metric_label(metric):
            if metric=='macro_f1':return 'Macro-F1'
            label=int(metric.removeprefix('recall_class_'))
            if thresholds and len(thresholds)==1 and label in (0,1):return f"Recall {'<' if label==0 else '>='} {thresholds[0]:.3g}s"
            return f'Recall class {label}'
        title=[metric_label(metric) for metric in metrics]
        write_latex(dest/'classification_table.tex',['Metric (%)',*columns],[[label,*row] for label,row in zip(title,values)])
    regression=table[table.task=='regression']
    if not regression.empty:
        write_latex(dest/'regression_table.tex',['Model','Role','N','MAE (s)','RMSE (s)','Median Q-error','Overestimate fraction'],
                    [[r.model,r.role,int(r['count']),r.mae,r.rmse,r.median_q_error,r.overprediction_fraction] for _,r in regression.iterrows()])
    (dest/'comparison.json').write_text(json.dumps(dict(candidate_observations=len(reference),thresholds=thresholds,dataset_sha256=reference_hash,models=names,inputs=[str(p) for p in inputs]),indent=2)+'\n')
    return dict(output=str(dest),models=len(inputs),candidate_observations=len(reference))
