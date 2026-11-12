"""Static graph features with an in-memory cache; model inputs remain read-only."""
import json
import os
import pickle
from pathlib import Path


class FeatureUnpickler(pickle.Unpickler):
    """Resolve historical feature class names to the implementation in this package."""
    def find_class(self, module, name):
        if module == 'q_graph.graph_construction.feats.featurizer2' and name == 'Featurizer2':
            from .features.feats.featurizer2 import Featurizer2
            return Featurizer2
        return super().find_class(module, name)


def load_featurizer(path):
    # Only trusted artifacts: pickle can execute Python code.
    with Path(path).open('rb') as handle:
        return FeatureUnpickler(handle).load()


class GraphCache:
    def __init__(self, featurizer, jar=None):
        self.featurizer = featurizer
        self.jar = Path(jar or os.environ.get('QG_JAR', 'java/qpe/target/qpe-1.0-SNAPSHOT.jar')).resolve()
        self.cache = {}

    def __call__(self, item):
        from .features.qp.query_plan_path import QueryGraph
        text = item['query_text']
        key = (text, item.get('graph_json'))
        if key not in self.cache:
            if item.get('graph_json'):
                raw = json.loads(Path(item['graph_json']).read_text())
            else:
                import jpype
                if not self.jar.is_file():
                    raise ValueError(f'Query graph JAR missing: {self.jar}; build java/qpe or supply --jar')
                if not jpype.isJVMStarted():
                    jpype.startJVM(classpath=[str(self.jar)])
                raw = json.loads(str(jpype.JClass('com.org.ex_plan.QueryGraphVisitor2').QGCreate(text)))
            graph = QueryGraph(raw)
            graph.feature(self.featurizer)
            self.cache[key] = graph.to_dgl()
        return self.cache[key]
