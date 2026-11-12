from pathlib import Path
import pytest

@pytest.mark.integration
def test_java_graph(tmp_path):
    jar=Path('src/qpp/java/qpe/target/qpe-1.0-SNAPSHOT.jar').resolve()
    if not jar.exists(): pytest.skip('Build src/qpp/java/qpe first')
    from qpp.graphs import GraphCache
    from qpp.features.feats.featurizer2 import Featurizer2
    for key in ('pred','ent'): (tmp_path/key/'freq').mkdir(parents=True)
    config=dict(pred_stat_path=str(tmp_path/'pred'),ent_path=str(tmp_path/'ent'),jar=str(jar),scaling='None')
    provider=GraphCache(Featurizer2(**{k:v for k,v in config.items() if k!='jar'}), jar)
    g=provider(dict(query_text='SELECT * WHERE { ?s <urn:p> ?o }'))
    assert g.num_nodes()==1 and g.num_edges()>0
    assert g.ndata['node_features'].shape[1]>0
