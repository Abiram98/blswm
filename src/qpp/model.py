"""Load and serve classification or total-runtime regression CQPP checkpoints."""
import json
import math
import threading
from pathlib import Path
from .protocol import normalize, ROLES, VERSION


class Predictor:
    def __init__(self, model_dir, device='cpu', jar=None):
        import torch
        from .network import CQPP, dimensions
        from .graphs import GraphCache, load_featurizer
        self.directory = Path(model_dir).resolve()
        self.config = json.loads((self.directory / 'model_config.json').read_text())
        checkpoint = torch.load(self.directory / 'best_model.pt', map_location='cpu', weights_only=False)
        self.metadata = checkpoint.get('metadata', {})
        state = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
        self.config.update({k: self.metadata[k] for k in ('variant',) if k in self.metadata})
        self.dimensions = dimensions(state, self.config)
        loss_type = self.metadata.get('loss_type', self.config.get('loss_type', 'cross_entropy'))
        self.task = 'regression' if loss_type == 'rmse' else 'classification'
        self.thresholds = [float(t) for t in self.metadata.get('runtime_thresholds', self.config.get('runtime_thresholds', []))]
        if (any(not math.isfinite(t) or t <= 0 for t in self.thresholds)
                or self.thresholds != sorted(set(self.thresholds))):
            raise ValueError('Checkpoint defines invalid runtime thresholds')
        if self.task == 'regression':
            target = self.metadata.get('target', self.config.get('target'))
            units = self.metadata.get('target_units', self.config.get('target_units', 'seconds'))
            if self.dimensions['num_classes'] != 1 or target != 'observed_runtime' or units != 'seconds':
                raise ValueError('Regression CQPP requires one observed_runtime output in seconds')
        elif not self.thresholds or len(self.thresholds) + 1 != self.dimensions['num_classes']:
            raise ValueError('Checkpoint does not define valid interval classification thresholds')
        self.device = torch.device(device)
        self.model = CQPP(**self.dimensions).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.graphs = GraphCache(load_featurizer(self.directory / 'featurizer.pickle'), jar)
        self.normalizer = self.metadata.get('normalizer', {})
        if any(not math.isfinite(float(self.normalizer.get(key, 0.))) for key in ('elapsed_mean', 'elapsed_std')):
            raise ValueError('Invalid elapsed normalizer')
        self.lock = threading.Lock()

    def info(self):
        return dict(status='ok', protocol_version=VERSION, model_type='CQPP',
                    prediction_task=self.task,
                    prediction_unit='seconds' if self.task == 'regression' else 'class_index',
                    runtime_semantics='total_runtime' if self.task == 'regression' else 'runtime_interval',
                    output_dim=self.dimensions['num_classes'], runtime_thresholds=self.thresholds,
                    num_classes=len(self.thresholds)+1 if self.task == 'classification' else None,
                    model_variant=self.dimensions['variant'],
                    dimensions=self.dimensions, device=str(self.device),
                    supports_query_set_prediction=True, returns_per_query_predictions=True)

    def tensors(self, queries):
        import dgl
        import torch
        graphs = dgl.batch([self.graphs(q) for q in queries]).to(self.device)
        lengths = torch.tensor([len(queries)], dtype=torch.long, device=self.device)
        roles = torch.tensor([[ROLES[q['role']] for q in queries]], device=self.device)
        mean = float(self.normalizer.get('elapsed_mean', 0))
        std = max(float(self.normalizer.get('elapsed_std', 1)), 1e-12)
        elapsed = torch.tensor([[[(math.log1p(q['elapsed_at_decision']) - mean) / std]
                                 for q in queries]], device=self.device)
        return graphs, lengths, roles, elapsed

    def predict(self, request):
        import torch
        queries = normalize(request.get('queries'))
        with self.lock, torch.inference_mode():
            logits = self.model(*self.tensors(queries))[0]
            outputs = (logits if self.task == 'regression' else logits.softmax(-1)).cpu().tolist()
        predictions = []
        for q, values in zip(queries, outputs):
            if any(not math.isfinite(p) for p in values):
                raise ValueError('Model produced nonfinite predictions')
            item = dict(query_id=q['query_id'], input_index=q['input_index'], position=q['pos'], role=q['role'])
            if self.task == 'regression':
                item['predicted_runtime_seconds'] = values[0]
            else:
                cls = max(range(len(values)), key=values.__getitem__)
                item.update(predicted_class=cls, confidence=values[cls],
                            probabilities={str(i): p for i, p in enumerate(values)})
            predictions.append(item)
        return dict(self.info(), sample_id=request.get('sample_id'), predictions=predictions)


class TrainedPredictor:
    """Serve checkpoints produced by qpp fit and tune."""
    def __init__(self, model_dir, device='cpu', jar=None):
        import torch
        from .architectures import PerformanceModel
        from .graphs import GraphCache, load_featurizer
        self.directory=Path(model_dir).resolve()
        checkpoint=torch.load(self.directory/'best_model.pt',map_location='cpu',weights_only=False)
        self.dimensions=checkpoint['architecture'];self.kind=self.dimensions['model_type']
        self.thresholds=checkpoint['thresholds'];self.task=checkpoint['task']
        self.device=torch.device(device)
        self.model=PerformanceModel(**self.dimensions).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'],strict=True);self.model.eval()
        self.graphs=GraphCache(load_featurizer(self.directory/'featurizer.pickle'),jar)
        self.lock=threading.Lock()

    def info(self):
        joint=self.kind not in ('planrgcn','dwns')
        return dict(status='ok',protocol_version=VERSION,model_type=self.kind,
                    prediction_task=self.task,prediction_unit='seconds' if self.task=='regression' else 'class_index',
                    runtime_semantics='total_runtime' if self.task=='regression' else 'runtime_interval',
                    output_dim=1 if self.task=='regression' else len(self.thresholds)+1,
                    runtime_thresholds=self.thresholds,num_classes=len(self.thresholds)+1 if self.task=='classification' else None,
                    model_variant=self.kind,dimensions=self.dimensions,device=str(self.device),
                    supports_query_set_prediction=True,returns_per_query_predictions=joint)

    def predict(self, request):
        import dgl,torch
        queries=normalize(request.get('queries'))
        indices=list(range(len(queries)))
        if self.kind in ('planrgcn','dwns'):
            indices=[i for i,q in enumerate(queries) if q['role']=='candidate']
            if len(indices)!=1:raise ValueError('Candidate-only models require one candidate')
            if self.kind=='dwns' and indices[0]!=len(queries)-1:raise ValueError('DWNS candidate must follow running queries')
        with self.lock,torch.inference_mode():
            graph_queries=[queries[i] for i in indices] if self.kind=='planrgcn' else queries
            graphs=dgl.batch([self.graphs(q) for q in graph_queries]).to(self.device)
            logits=self.model(graphs,graph_queries)
            if self.kind!='planrgcn':logits=logits[indices]
            outputs=(logits if self.task=='regression' else logits.softmax(-1)).cpu().tolist()
        predictions=[]
        for index,values in zip(indices,outputs):
            if any(not math.isfinite(x) for x in values):raise ValueError('Model produced nonfinite predictions')
            q=queries[index];item=dict(query_id=q['query_id'],input_index=q['input_index'],position=q['pos'],role=q['role'])
            if self.task=='regression':item['predicted_runtime_seconds']=values[0]
            else:
                label=max(range(len(values)),key=values.__getitem__)
                item.update(predicted_class=label,confidence=values[label],probabilities={str(i):p for i,p in enumerate(values)})
            predictions.append(item)
        return dict(self.info(),sample_id=request.get('sample_id'),predictions=predictions)


def load_predictor(model_dir, device='cpu', jar=None):
    import torch
    checkpoint=torch.load(Path(model_dir)/'best_model.pt',map_location='cpu',weights_only=False)
    cls=TrainedPredictor if 'architecture' in checkpoint else Predictor
    return cls(model_dir,device,jar)
