"""RGCN query encoder and bidirectional CQPP sequence head.

Parameter names are stable so existing JointDWNS checkpoints load strictly.
"""
import dgl
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dgl.nn.pytorch import RelGraphConv


class QueryEncoder(nn.Module):
    def __init__(self, input_d, hidden_dim1, hidden_dim2, num_classes, dropout, max_relations):
        super().__init__()
        self.conv1 = RelGraphConv(input_d, hidden_dim1, max_relations, dropout=dropout)
        self.conv2 = RelGraphConv(hidden_dim1, hidden_dim2, max_relations, dropout=dropout)
        self.classify = nn.Linear(hidden_dim2, num_classes)

    def forward(self, graph):
        h = torch.relu(self.conv1(graph, graph.ndata['node_features'].float(), graph.edata['rel_type']))
        h = torch.relu(self.conv2(graph, h, graph.edata['rel_type']))
        with graph.local_scope():
            graph.ndata['embedding'] = h
            return dgl.mean_nodes(graph, 'embedding')


class CQPP(nn.Module):
    def __init__(self, input_d, hidden_dim1=512, hidden_dim2=256, num_classes=2,
                 lstm_hidden_dim=128, lstm_num_layers=1, dropout=.1,
                 variant='joint_dwns_base', role_embedding_dim=8,
                 use_elapsed_at_decision=False, max_relations=13):
        super().__init__()
        if variant not in ('joint_dwns_base', 'joint_dwns_state'):
            raise ValueError(f'Unsupported CQPP variant: {variant}')
        self.variant = variant
        self.use_elapsed = use_elapsed_at_decision or variant == 'joint_dwns_state'
        self.query_encoder = QueryEncoder(input_d, hidden_dim1, hidden_dim2, num_classes, dropout, max_relations)
        self.role_embedding = nn.Embedding(3, role_embedding_dim) if variant == 'joint_dwns_state' else None
        extra = role_embedding_dim + int(self.use_elapsed) if self.role_embedding is not None else 0
        self.workload_encoder = nn.LSTM(hidden_dim2 + extra, lstm_hidden_dim, lstm_num_layers,
                                       batch_first=True, bidirectional=True,
                                       dropout=dropout if lstm_num_layers > 1 else 0.)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * lstm_hidden_dim, num_classes)

    def forward(self, graph, lengths, role_ids=None, elapsed=None):
        flat = self.query_encoder(graph)
        sequences = flat.new_zeros((len(lengths), int(lengths.max()), flat.shape[-1]))
        offset = 0
        for i, length in enumerate(lengths.tolist()):
            sequences[i, :length] = flat[offset:offset + length]
            offset += length
        if offset != flat.shape[0]:
            raise ValueError('Graph count does not match sequence lengths')
        if self.role_embedding is not None:
            pieces = [sequences, self.role_embedding(role_ids)]
            if self.use_elapsed:
                pieces.append(elapsed)
            sequences = torch.cat(pieces, dim=-1)
        packed = pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.workload_encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=sequences.shape[1])
        return self.classifier(self.dropout(encoded))


def dimensions(state, config):
    """Tuned checkpoints may disagree with their original configuration."""
    layers = 0
    while f'workload_encoder.weight_ih_l{layers}' in state:
        layers += 1
    return dict(
        input_d=int(state['query_encoder.conv1.loop_weight'].shape[0]),
        hidden_dim1=int(state['query_encoder.conv1.loop_weight'].shape[1]),
        hidden_dim2=int(state['query_encoder.conv2.loop_weight'].shape[1]),
        num_classes=int(state['classifier.weight'].shape[0]),
        lstm_hidden_dim=int(state['workload_encoder.weight_hh_l0'].shape[1]),
        lstm_num_layers=layers,
        max_relations=int(state['query_encoder.conv1.linear_r.W'].shape[0]),
        dropout=float(config.get('dropout', .1)),
        variant=config.get('variant', 'joint_dwns_base'),
        role_embedding_dim=int(state['role_embedding.weight'].shape[1]) if 'role_embedding.weight' in state else 8,
        use_elapsed_at_decision=bool(config.get('use_elapsed_at_decision', False)),
    )
