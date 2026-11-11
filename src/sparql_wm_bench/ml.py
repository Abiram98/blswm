"""Five PlanRGCN-based heads, one trainer and one checkpoint schema.

RGCN embedding follows sqpp Classifier2RGCN; joint encoder follows JointDWNS.
P+IconQ and joint RMSE are specification-derived additions.
"""
import time
import json
import numpy as np
from .io import read, write, output, digest
from .dataset import validate

MODELS=('planrgcn_class','scqpp_class','bi_scqpp_class','bi_scqpp_reg','p_iconq_class')

def network(config):
    import torch
    from torch import nn
    import dgl
    from dgl.nn.pytorch import RelGraphConv
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind=config['model_type']; h=config.get('hidden_dim2',config.get('hidden_dim',128)); h1=config.get('hidden_dim1',h)
            self.conv1=RelGraphConv(config['in_dim'],h1,13,dropout=config.get('dropout',.1))
            self.conv2=RelGraphConv(h1,h,13,dropout=config.get('dropout',.1))
            bidir=self.kind!='scqpp_class'
            self.lstm=nn.LSTM(2*h+3 if self.kind=='p_iconq_class' else h,h,batch_first=True,bidirectional=bidir)
            self.head=nn.Linear(h*(2 if bidir else 1),1 if self.kind.endswith('_reg') else 2)
            self.isolated_head=nn.Linear(h,2)
            self.dropout=nn.Dropout(config.get('dropout',.1))
        def forward(self, graphs, admissions):
            g=dgl.batch(graphs)
            z=torch.relu(self.conv1(g,g.ndata['node_features'].float(),g.edata['rel_type']))
            z=torch.relu(self.conv2(g,z,g.edata['rel_type']))
            with g.local_scope():
                g.ndata['z']=z; z=dgl.mean_nodes(g,'z')
            if self.kind=='planrgcn_class': return self.isolated_head(z)
            if self.kind=='p_iconq_class':
                rows=[]; t=torch.tensor(admissions,device=z.device)
                order=torch.argsort(t,stable=True)
                for j in range(len(graphs)):
                    features=torch.cat((z[j].expand_as(z),z,abs(t-t[j])[:,None],(t<t[j]).float()[:,None],(t>t[j]).float()[:,None]),dim=1)
                    enc,_=self.lstm(features[order][None]); pos=(order==j).nonzero()[0,0]
                    rows.append(self.head(self.dropout(enc[0,pos])))
                return torch.stack(rows)
            enc,_=self.lstm(z[None]); return self.head(self.dropout(enc[0]))
    if config['model_type'] not in MODELS: raise ValueError('Unknown model')
    return Model()

def metadata(config):
    return dict(schema_version=1,architecture=config,feature_schema='sqpp-Featurizer2-QueryGraph-13',
                thresholds=[config.get('class_threshold_s',120)],units='seconds' if config['model_type'].endswith('_reg') else 'class',
                semantics='total_execution_runtime',normalization=None)

def fit(config, train, validation, graph_provider, folder):
    import torch
    torch.manual_seed(config.get('seed',1)); np.random.seed(config.get('seed',1))
    device=config.get('device','cpu'); model=network(config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.get('learning_rate',.001))
    regression=config['model_type'].endswith('_reg'); start=time.perf_counter()
    def loss(sample):
        validate(sample)
        graphs=[graph_provider(x).to(device) for x in sample['S']]
        pred=model(graphs,[x['admission_s'] for x in sample['S']])
        mask=torch.tensor([config['model_type'] not in ('scqpp_class','planrgcn_class') or x['role']=='candidate' for x in sample['S']],device=device)
        if not mask.any(): return None
        target=torch.tensor(sample['y'],device=device,dtype=torch.float32)
        return torch.sqrt(torch.mean((pred[:,0][mask]-target[mask])**2)+1e-12) if regression else torch.nn.functional.cross_entropy(pred[mask],(target[mask]>=config.get('class_threshold_s',120)).long())
    history=[]; best=float('inf')
    for epoch in range(config.get('epochs',10)):
        model.train(); losses=[]
        for sample in train:
            optimizer.zero_grad(); value=loss(sample)
            if value is None: continue
            value.backward(); optimizer.step(); losses.append(value.item())
        model.eval()
        with torch.no_grad(): vals=[v.item() for s in validation if (v:=loss(s)) is not None]
        if not vals or not losses: raise ValueError('No supervised samples')
        score=float(np.mean(vals)); history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_loss=score))
        if score<best:
            best=score; torch.save(model.state_dict(),output(f'{folder}/weights.pt'))
    meta=metadata(config); meta.update(training_time_s=time.perf_counter()-start,train_hash=digest(train),validation_hash=digest(validation))
    if hasattr(graph_provider,'featurizer'):
        import pickle,hashlib
        data=pickle.dumps(graph_provider.featurizer)
        output(f'{folder}/features.pickle').write_bytes(data)
        meta['feature_hash']=hashlib.sha256(data).hexdigest()
        meta['graph_config']=graph_provider.config
        meta['normalization']='fitted scalers in features.pickle'
    else: meta['feature_source']='injected test graphs; not a paper checkpoint'
    write(f'{folder}/metadata.json',meta); write(f'{folder}/history.json',history)
    return meta

class Loaded:
    def __init__(self, folder, graph_provider, device='cpu'):
        import torch
        self.meta=read(f'{folder}/metadata.json'); self.config=self.meta['architecture']
        expected=metadata(self.config)
        for k in ('schema_version','thresholds','units','semantics','feature_schema'):
            if self.meta[k]!=expected[k]: raise ValueError(f'Checkpoint metadata mismatch: {k}')
        self.model=network(self.config).to(device)
        self.model.load_state_dict(torch.load(f'{folder}/weights.pt',map_location=device,weights_only=True)); self.model.eval()
        if self.meta.get('feature_hash'):
            import pickle,hashlib
            from pathlib import Path
            data=Path(f'{folder}/features.pickle').read_bytes()
            if hashlib.sha256(data).hexdigest()!=self.meta['feature_hash']: raise ValueError('Feature artifact hash mismatch')
            graph_provider.featurizer=pickle.loads(data)
            graph_provider.cache.clear()
        self.graph_provider,self.device=graph_provider,device
    def state(self, items):
        import torch
        with torch.no_grad():
            pred=self.model([self.graph_provider(x).to(self.device) for x in items],[x['admission_s'] for x in items])
            values=pred[:,0] if self.config['model_type'].endswith('_reg') else pred.argmax(1)
            values=values.cpu().tolist()
        if not np.isfinite(values).all(): raise ValueError('Nonfinite model output')
        return {x['occurrence_id']:v for x,v in zip(items,values)}

class Online:
    def __init__(self, loaded, catalog):
        self.loaded=loaded; self.texts={r['query_id']:r['query_text'] for r in catalog}
    def predict_joint(self,candidate,running,now):
        items=[dict(occurrence_id=r.occurrence.occurrence_id,query_text=self.texts[r.occurrence.query_id],admission_s=r.admission_s) for r in running]
        items.append(dict(occurrence_id=candidate.occurrence_id,query_text=self.texts[candidate.query_id],admission_s=now))
        return self.loaded.state(items)
    def predict(self,candidate,running,now):
        if self.loaded.config['model_type']=='planrgcn_class': running=()
        return self.predict_joint(candidate,running,now)[candidate.occurrence_id]

def evaluate(loaded, samples):
    from .analysis import prediction_metrics
    labels=[]; predictions=[]; roles=[]; elapsed=[]
    for s in samples:
        validate(s)
        import torch
        if loaded.device.startswith('cuda'): torch.cuda.synchronize()
        start=time.perf_counter(); result=loaded.state(s['S'])
        if loaded.device.startswith('cuda'): torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
        for item,y in zip(s['S'],s['y']):
            if loaded.config['model_type'] in ('scqpp_class','planrgcn_class') and item['role']!='candidate': continue
            labels.append(y if loaded.config['model_type'].endswith('_reg') else int(y>=loaded.config.get('class_threshold_s',120)))
            predictions.append(result[item['occurrence_id']]); roles.append(item['role'])
    task='regression' if loaded.config['model_type'].endswith('_reg') else 'classification'
    out=prediction_metrics(labels,predictions,task)
    out['roles']={role:prediction_metrics([y for y,r in zip(labels,roles) if r==role],[p for p,r in zip(predictions,roles) if r==role],task) for role in set(roles)}
    out['inference_s']=elapsed
    out['timing_protocol']='local end-to-end state inference including lazy graph construction; excludes model load; synchronized CUDA'
    return out

def benchmark(loaded, items, repeats=10):
    """Warm graph cache; time repeated state inference, excluding model load."""
    import torch
    if len(items)<60: raise ValueError('Timing matrix needs at least 60 query items')
    def sync():
        if loaded.device.startswith('cuda'): torch.cuda.synchronize()
    graph_start=time.perf_counter()
    for item in items: loaded.graph_provider(item)
    sync(); graph_s=time.perf_counter()-graph_start
    rows=[]
    for running in (0,5,10):
        for waiting in (2,5,10,20,50):
            elapsed=[]
            for _ in range(repeats):
                sync(); start=time.perf_counter()
                for i in range(waiting): loaded.state(items[:running]+[items[running+i]])
                sync(); elapsed.append(time.perf_counter()-start)
            rows.append(dict(running=running,waiting=waiting,repeats=repeats,mean_s=float(np.mean(elapsed)),samples_s=elapsed))
    return dict(protocol='warm graph cache; one prospective state per candidate; synchronized CUDA; excludes load and network',graph_warmup_s=graph_s,rows=rows)
