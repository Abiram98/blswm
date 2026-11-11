"""Portable adapter around the sqpp logical graph and Meta-KG featurizer."""
import json
from pathlib import Path

class Graphs:
    def __init__(self, config):
        from .vendor.feats.featurizer2 import Featurizer2
        self.config=config
        for key in ('pred_stat_path','ent_path'):
            if not Path(config[key]).exists(): raise ValueError(f'Missing Meta-KG input: {key}')
        self.featurizer=Featurizer2(**{k:v for k,v in config.items() if k in ('pred_stat_path','ent_path','lit_path','bins','scaling')})
        self.cache={}
    def __call__(self, item):
        from .vendor.qp.query_plan_path import QueryGraph
        text=item['query_text']
        if text not in self.cache:
            if item.get('graph_json'):
                raw=json.loads(Path(item['graph_json']).read_text())
            else:
                import jpype
                jar=Path(self.config['jar']).resolve()
                if not jar.is_file(): raise ValueError('Build java/qpe or configure an existing query-graph JAR')
                if not jpype.isJVMStarted(): jpype.startJVM(classpath=[str(jar)])
                raw=json.loads(str(jpype.JClass('com.org.ex_plan.QueryGraphVisitor2').QGCreate(text)))
            q=QueryGraph(raw); q.feature(self.featurizer); self.cache[text]=q.to_dgl()
        return self.cache[text]
