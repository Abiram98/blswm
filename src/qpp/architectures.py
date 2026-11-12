"""PlanRGCN encoders and candidate or all-query concurrency heads."""
import torch
from torch import nn
from .network import QueryEncoder

MODEL_TYPES=('planrgcn','dwns','qsps','qsps-regression','p-iconq')


class PerformanceModel(nn.Module):
    def __init__(self, model_type, input_d, hidden_dim1=512, hidden_dim2=128, lstm_hidden_dim=128,
                 lstm_num_layers=1, dropout=.1, num_classes=2, lstm_dropout=None):
        super().__init__()
        if model_type not in MODEL_TYPES:raise ValueError('Unsupported model type')
        self.model_type=model_type
        self.encoder=QueryEncoder(input_d,hidden_dim1,hidden_dim2,num_classes,dropout,13)
        self.dropout=nn.Dropout(dropout)
        if model_type=='planrgcn':
            self.classifier=nn.Linear(hidden_dim2,num_classes)
        else:
            bidirectional=model_type!='dwns'
            width=2*hidden_dim2+3 if model_type=='p-iconq' else hidden_dim2
            self.lstm=nn.LSTM(width,lstm_hidden_dim,lstm_num_layers,batch_first=True,
                              bidirectional=bidirectional,dropout=(dropout if lstm_dropout is None else lstm_dropout) if lstm_num_layers>1 else 0.)
            output=1 if model_type=='qsps-regression' else num_classes
            self.classifier=nn.Linear(lstm_hidden_dim*(2 if bidirectional else 1),output)

    def forward(self, graph, queries):
        z=self.encoder(graph)
        if len(queries)!=len(z):raise ValueError('Graph and state sizes differ')
        if self.model_type=='planrgcn':return self.classifier(self.dropout(z))
        if self.model_type=='p-iconq':
            raw_times=([float(q['start_time']) for q in queries] if all(q.get('start_time') is not None for q in queries)
                       else [-float(q.get('elapsed_at_decision',0)) for q in queries])
            origin=min(raw_times)
            times=torch.tensor([t-origin for t in raw_times],device=z.device,dtype=z.dtype)
            order=torch.argsort(times,stable=True); outputs=[]
            for index in range(len(queries)):
                delta=times-times[index]
                x=torch.cat((z[index].expand_as(z),z,delta.abs()[:,None],(delta<0).float()[:,None],(delta>0).float()[:,None]),dim=1)
                encoded,_=self.lstm(x[order].unsqueeze(0))
                position=(order==index).nonzero(as_tuple=True)[0][0]
                outputs.append(self.classifier(self.dropout(encoded[0,position])))
            return torch.stack(outputs)
        encoded,_=self.lstm(z.unsqueeze(0))
        return self.classifier(self.dropout(encoded[0]))
