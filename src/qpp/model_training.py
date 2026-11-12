"""Train self-contained query-performance models from CSV or query-set JSONL."""
import ast
import csv
import hashlib
import json
import math
import random
import pickle
import shutil
import sys
from pathlib import Path

from .protocol import normalize
from .graphs import GraphCache, load_featurizer
from .training import read_states
from wrk_gen.queries import new_directory, fingerprint


def make_featurizer(predicate_stats, entity_stats, literal_stats, output, *, bins=10, scaling='binner'):
    """Fit and save graph features from local knowledge-graph statistics."""
    if type(bins) is not int or bins<2 or scaling not in ('binner','none'):
        raise ValueError('Invalid feature bins or scaling')
    if not predicate_stats or not entity_stats:
        raise ValueError('Predicate and entity statistics are required')
    sources=[Path(value).expanduser().resolve() for value in (predicate_stats,entity_stats,literal_stats) if value]
    if any(not (source/'freq').is_dir() for source in sources):
        raise ValueError('Each statistics directory must contain a freq directory')
    if len(set(sources))!=len(sources):raise ValueError('Statistics directories must be distinct')
    if scaling=='binner':
        needed=[sources[0]/name for name in ('freq','ents','lits')]
        needed.extend(sources[1]/name for name in ('freq','subj','obj'))
        if literal_stats:needed.append(sources[2]/'freq')
        if any(not directory.is_dir() or not any(directory.glob('*.json')) for directory in needed):
            raise ValueError('Binned features require JSON files in predicate freq/ents/lits and entity freq/subj/obj directories, plus literal freq when used')
    hashes={str(path):fingerprint(path) for source in sources for path in sorted(source.rglob('*.json')) if path.is_file()}
    if not hashes:raise ValueError('Statistics directories contain no JSON files')
    from .features.feats.featurizer2 import Featurizer2
    feat=Featurizer2(pred_stat_path=str(sources[0]),ent_path=str(sources[1]),
                     lit_path=str(sources[2]) if literal_stats else None,bins=bins,
                     scaling='None' if scaling=='none' else 'binner')
    if any(fingerprint(path)!=digest for path,digest in hashes.items()):raise ValueError('Statistics changed during featurizer fitting')
    dest=new_directory(output,sources)
    with (dest/'featurizer.pickle').open('wb') as stream:pickle.dump(feat,stream,protocol=pickle.HIGHEST_PROTOCOL)
    result=dict(output=str(dest),featurizer=str(dest/'featurizer.pickle'),bins=bins,scaling=scaling,
                input_sha256=hashes,feature_dimension=int(feat.base_size+feat.path_size+feat.filter_size))
    (dest/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result



def validate_config(config):
    from .architectures import MODEL_TYPES
    integers=('hidden_dim1','hidden_dim2','lstm_hidden_dim','lstm_num_layers','epochs','threads','batch_size')
    if config['model_type'] not in MODEL_TYPES or any(type(config[key]) is not int or config[key]<1 for key in integers):
        raise ValueError('Invalid model type or positive integer setting')
    if type(config['seed']) is not int or config['seed']<0 or config['optimizer'] not in ('adam','adamw'):
        raise ValueError('Invalid seed or optimizer')
    for key in ('dropout','label_smoothing','lr','weight_decay','grad_clip'):
        value=config[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if not 0<=config['dropout']<1 or not 0<=config['label_smoothing']<1 or config['lr']<=0 or config['weight_decay']<0 or config['grad_clip']<=0:
        raise ValueError('Invalid optimizer or loss setting')
    lstm_dropout=config.get('lstm_dropout')
    if lstm_dropout is not None and (isinstance(lstm_dropout,bool) or not isinstance(lstm_dropout,(int,float)) or not math.isfinite(lstm_dropout) or not 0<=lstm_dropout<1):
        raise ValueError('Invalid LSTM dropout')
    if config.get('init_checkpoint') and config.get('encoder_checkpoint'):
        raise ValueError('Use either init_checkpoint or encoder_checkpoint')
    if config.get('selection_metric','loss') not in ('loss','macro_f1') or config['model_type']=='qsps-regression' and config.get('selection_metric')=='macro_f1':
        raise ValueError('Macro-F1 selection requires a classification model')
    bins=config.get('feature_bins');scaling=config.get('feature_scaling')
    if bins is not None and (type(bins) is not int or bins<2):raise ValueError('feature_bins must be an integer >= 2')
    if scaling is not None and scaling not in ('binner','none'):raise ValueError('Invalid feature_scaling')
    thresholds=config['thresholds']
    if not thresholds or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in thresholds) or thresholds!=sorted(set(thresholds)):
        raise ValueError('Thresholds must be sorted, unique, positive and finite')

def load_data(path, model_type, thresholds):
    path=Path(path).resolve()
    if path.suffix=='.jsonl':
        rows=read_states(path,len(thresholds)+1 if model_type=='qsps-regression' else 2**31,
                         'regression' if model_type=='qsps-regression' else 'classification')
        if model_type!='qsps-regression':
            # Derive classes from the requested thresholds when runtimes are present.
            for row in rows:
                if all('observed_runtime' in q for q in row['queries']):
                    runtimes=[q['observed_runtime'] for q in row['queries']]
                    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in runtimes):
                        raise ValueError('Observed runtimes must be finite and nonnegative')
                    row['labels']=[sum(v>=t for t in thresholds) for v in runtimes]
                elif any(y>=len(thresholds)+1 for y in row['labels']):
                    raise ValueError('Labels exceed requested intervals and observed runtimes are unavailable')
        return rows
    if model_type not in ('planrgcn','dwns'):
        raise ValueError('Joint models require query-set JSONL data')
    rows=[]
    csv.field_size_limit(sys.maxsize)
    with path.open(newline='',encoding='utf-8-sig') as stream:
        for index,row in enumerate(csv.DictReader(stream,delimiter='\t' if path.suffix=='.tsv' else ','),1):
            text=row.get('query_text') or row.get('queryString')
            raw=row.get('Execution_Time') or row.get('mean_latency')
            try:runtime=float(raw)
            except (ValueError,TypeError) as exc:raise ValueError(f'{path}:{index}: missing runtime') from exc
            if not math.isfinite(runtime) or runtime<0 or not text:raise ValueError(f'{path}:{index}: invalid query or runtime')
            try: running=[] if model_type=='planrgcn' else ast.literal_eval(row.get('Running_Queries_At_Start') or '[]')
            except (ValueError,SyntaxError) as exc:raise ValueError(f'{path}:{index}: invalid running context') from exc
            if not isinstance(running,list):raise ValueError('Running context must be a list')
            queries=[]
            for entry in running:
                if isinstance(entry,str):entry={'text':entry}
                if not isinstance(entry,dict):raise ValueError('Invalid running query')
                item=entry.get('query_text') or entry.get('text')
                if not item:raise ValueError('Running query has no text')
                queries.append(dict(query_text=item,query_id=str(entry.get('query_id','running')),role='running',elapsed_at_decision=0))
            queries.append(dict(query_text=text,query_id=str(row.get('Query_ID') or row.get('id') or index),role='candidate',elapsed_at_decision=0))
            rows.append(dict(queries=normalize(queries),labels=[0]*(len(queries)-1)+[sum(runtime>=t for t in thresholds)],sample_id=str(index),split_group_id=None))
    if not rows:raise ValueError(f'Empty dataset: {path}')
    return rows


def fit(config, destination, *, report=None):
    validate_config(config)
    import torch,dgl
    from .architectures import PerformanceModel
    torch.set_num_threads(config['threads']);torch.manual_seed(config['seed'])
    rng=random.Random(config['seed'])
    kind=config['model_type'];regression=kind=='qsps-regression'
    inputs=[config['data'],config['validation'],config['featurizer'],*([config['init_checkpoint']] if config.get('init_checkpoint') else []),*([config['encoder_checkpoint']] if config.get('encoder_checkpoint') else [])]
    hashes={str(Path(path).resolve()):fingerprint(path) for path in inputs}
    thresholds=config['thresholds'];device=torch.device(config['device'])
    train=load_data(config['data'],kind,thresholds);validation=load_data(config['validation'],kind,thresholds)
    groups={r['split_group_id'] for r in train if r['split_group_id'] is not None}
    if groups & {r['split_group_id'] for r in validation if r['split_group_id'] is not None}:
        raise ValueError('Training and validation split groups overlap')
    if Path(config['data']).resolve()==Path(config['validation']).resolve():raise ValueError('Training and validation inputs must differ')
    features=Path(config['featurizer']).resolve()
    featurizer=load_featurizer(features)
    refit=config.get('feature_bins') is not None or config.get('feature_scaling') is not None
    if refit:
        if config.get('feature_bins') is not None:featurizer.bins=config['feature_bins']
        if config.get('feature_scaling') is not None:featurizer.scaling={'none':'None','binner':'binner'}[config['feature_scaling']]
        featurizer._initialize_scalers();featurizer._calculate_sizes()
        featurizer.tp_mask=[1]*featurizer.base_size+[0]*featurizer.path_size+[0]*featurizer.filter_size
        featurizer.pp_mask=[1]*featurizer.base_size+[1]*featurizer.path_size+[0]*featurizer.filter_size
        featurizer.filter_mask=[0]*featurizer.base_size+[0]*featurizer.path_size+[1]*featurizer.filter_size
        featurizer.tp_size=featurizer.base_size+featurizer.path_size+featurizer.filter_size
    provider=GraphCache(featurizer,config.get('jar'))
    input_d=int(provider(train[0]['queries'][0]).ndata['node_features'].shape[1])
    settings=dict(model_type=kind,input_d=input_d,hidden_dim1=config['hidden_dim1'],hidden_dim2=config['hidden_dim2'],
                  lstm_hidden_dim=config['lstm_hidden_dim'],lstm_num_layers=config['lstm_num_layers'],
                  dropout=config['dropout'],num_classes=len(thresholds)+1,lstm_dropout=config.get('lstm_dropout'))
    model=PerformanceModel(**settings).to(device)
    if config.get('init_checkpoint'):
        checkpoint=torch.load(config['init_checkpoint'],map_location='cpu',weights_only=False)
        if 'architecture' in checkpoint and checkpoint['architecture']['model_type']!=kind:
            raise ValueError('Initialization checkpoint has a different model type')
        recorded=checkpoint.get('thresholds') or checkpoint.get('metadata',{}).get('runtime_thresholds')
        if not regression and recorded is not None and list(recorded)!=thresholds:
            raise ValueError('Initialization checkpoint runtime thresholds do not match')
        state=checkpoint.get('model_state',checkpoint.get('state_dict',checkpoint))
        if not isinstance(state,dict):raise ValueError('Invalid initialization checkpoint')
        mapped={}
        for key,value in state.items():
            if key.startswith('query_encoder.'):key='encoder.'+key[len('query_encoder.'):]
            elif key.startswith('workload_encoder.'):key='lstm.'+key[len('workload_encoder.'):]
            elif key.startswith('rgcn.'):key='encoder.'+key[len('rgcn.'):]
            elif key.startswith(('conv1.','conv2.')):key='encoder.'+key
            elif key.startswith('classify.'):key='classifier.'+key[len('classify.'):]
            mapped[key]=value
        expected=model.state_dict()
        required={key for key in expected if not key.startswith('encoder.classify.')}
        missing=required-mapped.keys()
        bad=[key for key in required & mapped.keys() if expected[key].shape!=mapped[key].shape]
        if missing or bad:
            raise ValueError(f'Initialization architecture mismatch: missing={sorted(missing)}, shape_mismatch={sorted(bad)}')
        compatible={key:value for key,value in mapped.items() if key in expected and expected[key].shape==value.shape}
        model.load_state_dict(compatible,strict=False)
    if config.get('encoder_checkpoint'):
        checkpoint=torch.load(config['encoder_checkpoint'],map_location='cpu',weights_only=False)
        state=checkpoint.get('model_state',checkpoint.get('state_dict',checkpoint))
        mapped={}
        for key,value in state.items():
            for prefix in ('query_encoder.','rgcn.','encoder.'):
                if key.startswith(prefix):key=key[len(prefix):];break
            mapped[key]=value
        expected=model.encoder.state_dict()
        required={key for key in expected if not key.startswith('classify.')}
        missing=required-mapped.keys()
        bad=[key for key in required & mapped.keys() if expected[key].shape!=mapped[key].shape]
        if missing or bad:raise ValueError(f'Encoder architecture mismatch: missing={sorted(missing)}, shape_mismatch={sorted(bad)}')
        model.encoder.load_state_dict({key:value for key,value in mapped.items() if key in expected and expected[key].shape==value.shape},strict=False)
    if config.get('freeze_query_encoder'):
        model.encoder.requires_grad_(False)
    optimizer_cls=torch.optim.AdamW if config['optimizer']=='adamw' else torch.optim.Adam
    optimizer=optimizer_cls(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    best=float('inf');best_loss=None;best_f1=None;history=[];destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    if refit:
        with (destination/'featurizer.pickle').open('wb') as stream:pickle.dump(featurizer,stream,protocol=pickle.HIGHEST_PROTOCOL)
    else:shutil.copyfile(features,destination/'featurizer.pickle')
    def execute(row):
        queries=row['queries']
        indices=[i for i,q in enumerate(queries) if q['role']=='candidate'] if kind in ('dwns','planrgcn') else list(range(len(queries)))
        if not indices:raise ValueError('Candidate-only model needs a candidate in each state')
        if kind=='dwns' and (len(indices)!=1 or indices[0]!=len(queries)-1):
            raise ValueError('DWNS needs one candidate after running queries')
        graph_queries=[queries[i] for i in indices] if kind=='planrgcn' else queries
        graphs=dgl.batch([provider(q) for q in graph_queries]).to(device)
        logits=model(graphs,graph_queries)
        if kind!='planrgcn':logits=logits[indices]
        target=torch.tensor([row['labels'][i] for i in indices],dtype=torch.float32 if regression else torch.long,device=device)
        loss=torch.sqrt(torch.nn.functional.mse_loss(logits[:,0],target)+1e-8) if regression else torch.nn.functional.cross_entropy(logits,target,label_smoothing=config['label_smoothing'])
        return loss,logits,target
    for epoch in range(1,config['epochs']+1):
        model.train()
        if config.get('freeze_query_encoder'):model.encoder.eval()
        order=list(range(len(train)));rng.shuffle(order);losses=[]
        for offset in range(0,len(order),config['batch_size']):
            batch=order[offset:offset+config['batch_size']]
            optimizer.zero_grad()
            for index in batch:
                loss,_,_=execute(train[index])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss/len(batch)).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip']);optimizer.step()
        model.eval();validation_losses=[];truth=[];predicted=[]
        with torch.inference_mode():
            for item in validation:
                loss,logits,target=execute(item)
                validation_losses.append(float(loss))
                if not regression:
                    truth.extend(target.cpu().tolist());predicted.extend(logits.argmax(-1).cpu().tolist())
        validation_loss=sum(validation_losses)/len(validation_losses)
        if not math.isfinite(validation_loss):raise ValueError('Nonfinite validation loss')
        macro_f1=None
        if not regression:
            f1=[]
            for label in range(len(thresholds)+1):
                tp=sum(y==label and p==label for y,p in zip(truth,predicted))
                denominator=sum(y==label for y in truth)+sum(p==label for p in predicted)
                if denominator:f1.append(2*tp/denominator)
            macro_f1=sum(f1)/len(f1)
        score=1-macro_f1 if config['selection_metric']=='macro_f1' else validation_loss
        row=dict(epoch=epoch,train_loss=sum(losses)/len(losses),validation_loss=validation_loss,
                 validation_macro_f1=macro_f1,selection_score=score);history.append(row)
        if score<best:
            best=score;best_loss=validation_loss;best_f1=macro_f1
            torch.save(dict(model_state=model.state_dict(),architecture=settings,thresholds=thresholds,
                            task='regression' if regression else 'classification'),destination/'best_model.pt')
        if report:report(row)
    if any(fingerprint(path)!=digest for path,digest in hashes.items()):raise ValueError('Training input changed during execution')
    result=dict(model_type=kind,selection_metric=config['selection_metric'],best_selection_score=best,
                best_validation_loss=best_loss,best_validation_macro_f1=best_f1,epochs=len(history),architecture=settings,
                thresholds=thresholds,config=config,input_sha256=hashes)
    (destination/'model_config.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    return result


def tune_models(config, destination):
    """Run independent trials under Ray Tune and retain the best checkpoint."""
    import ray
    from ray import tune, train
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.basic_variant import BasicVariantGenerator
    path=Path(config['search']).resolve()
    search_hash=fingerprint(path)
    search=json.loads(path.read_text())
    allowed={'lr','weight_decay','dropout','hidden_dim1','hidden_dim2','lstm_hidden_dim','lstm_num_layers','grad_clip','batch_size','optimizer','label_smoothing','feature_bins','feature_scaling','lstm_dropout'}
    if not isinstance(search,dict) or not search or set(search)-allowed or any(not isinstance(v,list) or not v for v in search.values()):
        raise ValueError('Search JSON must map supported hyperparameters to nonempty value lists')
    for key,values in search.items():
        for value in values:validate_config(dict(config,**{key:value}))
    if min(config['samples'],config['cpus'],config['max_concurrent_trials'])<1 or config['cpus_per_trial']<=0 or config['gpus_per_trial']<0 or config['gpus']<config['gpus_per_trial']:
        raise ValueError('Invalid Ray resource or trial settings')
    dest=new_directory(destination)
    base=dict(config)
    def trainable(parameters):
        import torch
        trial=dict(base,**parameters)
        trial_dir=Path(tune.get_context().get_trial_dir())/'model'
        best=[float('inf')];selected=[None];history=[]
        source_paths=(trial['data'],trial['validation'],trial['featurizer'],*([trial['init_checkpoint']] if trial.get('init_checkpoint') else []),*([trial['encoder_checkpoint']] if trial.get('encoder_checkpoint') else []))
        source_hashes={str(Path(path).resolve()):fingerprint(path) for path in source_paths}
        def report(metrics):
            history.append(metrics)
            if metrics['selection_score']<best[0]:
                best[0]=metrics['selection_score'];selected[0]=metrics
            checkpoint=torch.load(trial_dir/'best_model.pt',map_location='cpu',weights_only=False)
            summary=dict(model_type=trial['model_type'],selection_metric=trial['selection_metric'],best_selection_score=best[0],
                         best_validation_loss=selected[0]['validation_loss'],best_validation_macro_f1=selected[0]['validation_macro_f1'],epochs=len(history),
                         architecture=checkpoint['architecture'],thresholds=trial['thresholds'],config=trial,
                         input_sha256=source_hashes)
            (trial_dir/'model_config.json').write_text(json.dumps(summary,indent=2)+'\n')
            (trial_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            tune.report(dict(metrics,best_selection_score=best[0]),checkpoint=tune.Checkpoint.from_directory(str(trial_dir)))
        fit(trial,trial_dir,report=report)
    ray.init(num_cpus=config['cpus'],num_gpus=config['gpus'],ignore_reinit_error=True,include_dashboard=False)
    try:
        tuner=tune.Tuner(tune.with_resources(trainable,{'cpu':config['cpus_per_trial'],'gpu':config['gpus_per_trial']}),
            param_space={key:tune.choice(value) for key,value in search.items()},
            tune_config=tune.TuneConfig(metric='best_selection_score',mode='min',num_samples=config['samples'],
                search_alg=BasicVariantGenerator(random_state=config['seed'],max_concurrent=config['max_concurrent_trials']),scheduler=ASHAScheduler(max_t=config['epochs'],grace_period=1)),
            run_config=tune.RunConfig(storage_path=str((dest/'ray').resolve()),name='search',verbose=0,
                checkpoint_config=tune.CheckpointConfig(num_to_keep=1,checkpoint_score_attribute='best_selection_score',checkpoint_score_order='min')))
        try:
            grid=tuner.fit()
        except Exception as exc:
            raise RuntimeError(f'Ray Tune failed; inspect trial logs in {dest / "ray"}: {exc}') from exc
        good=[r for r in grid if r.error is None and r.metrics.get('best_selection_score') is not None]
        if not good:raise RuntimeError('All Ray Tune trials failed; inspect trial logs')
        best=min(good,key=lambda r:r.metrics['best_selection_score'])
        if fingerprint(path)!=search_hash:raise ValueError('Search space changed during tuning')
        if best.checkpoint is None:raise RuntimeError('Best Ray trial produced no checkpoint')
        with best.checkpoint.as_directory() as checkpoint_dir:
            for name in ('best_model.pt','model_config.json','history.json','featurizer.pickle'):
                shutil.copyfile(Path(checkpoint_dir)/name,dest/name)
        result=dict(model_type=config['model_type'],selection_metric=config['selection_metric'],best_selection_score=best.metrics['best_selection_score'],
                    best_validation_loss=json.loads((dest/'model_config.json').read_text())['best_validation_loss'],
                    best_hyperparameters=best.config,successful_trials=len(good),requested_trials=config['samples'],
                    search_sha256=search_hash)
        (dest/'tuning.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:ray.shutdown()


def evaluate_model(predictor, path):
    """Score the supervised positions of a newly trained checkpoint."""
    rows=load_data(path,predictor.kind,predictor.thresholds)
    regression=predictor.task=='regression'
    predictions=[];truth=[];values=[]
    for row in rows:
        result=predictor.predict(dict(queries=row['queries'],sample_id=row['sample_id']))
        for prediction in result['predictions']:
            target=row['labels'][prediction['input_index']]
            truth.append(target)
            values.append(prediction['predicted_runtime_seconds'] if regression else prediction['predicted_class'])
        predictions.append(dict(sample_id=row['sample_id'],predictions=result['predictions']))
    if not truth:raise ValueError('No supervised predictions')
    if regression:
        errors=[a-b for a,b in zip(values,truth)]
        metrics=dict(rmse=math.sqrt(sum(x*x for x in errors)/len(errors)),mae=sum(abs(x) for x in errors)/len(errors))
    else:
        count=len(predictor.thresholds)+1
        matrix=[[sum(y==i and p==j for y,p in zip(truth,values)) for j in range(count)] for i in range(count)]
        f1=[2*matrix[i][i]/denominator for i in range(count)
            if (denominator:=sum(matrix[i])+sum(row[i] for row in matrix))]
        metrics=dict(accuracy=sum(a==b for a,b in zip(values,truth))/len(truth),macro_f1=sum(f1)/count,confusion_matrix=matrix)
    return dict(model_type=predictor.kind,prediction_task=predictor.task,states=len(rows),queries=len(truth),predictions=predictions,**metrics)
