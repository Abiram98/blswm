import pytest

@pytest.mark.parametrize('kind',['planrgcn_class','scqpp_class','bi_scqpp_class','bi_scqpp_reg','p_iconq_class'])
def test_train_reload(kind,tmp_path):
    import torch
    import dgl
    from sparql_wm_bench.ml import fit,Loaded
    torch.set_num_threads(1)
    def graph(item):
        g=dgl.graph(([0],[0])); g.ndata['node_features']=torch.tensor([[1.,2.,3.]])
        g.edata['rel_type']=torch.tensor([0]); return g
    samples=[dict(S=[dict(occurrence_id='r',query_text='running',admission_s=0,role='running'),dict(occurrence_id='c',query_text='candidate',admission_s=1,role='candidate')],y=[2.,140.])]
    cfg=dict(model_type=kind,in_dim=3,hidden_dim=4,epochs=2,seed=1)
    fit(cfg,samples,samples,graph,tmp_path)
    loaded=Loaded(tmp_path,graph)
    predictions=loaded.state(samples[0]['S'])
    assert list(predictions)==['r','c']
    assert all(torch.isfinite(torch.tensor(v)) for v in predictions.values())
    if kind=='bi_scqpp_reg': assert loaded.meta['units']=='seconds'
