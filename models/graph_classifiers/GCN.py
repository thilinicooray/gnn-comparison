import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import SAGEConv, global_mean_pool

class Dense(Module):
    """
    Simple Dense layer, Do not consider adj.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, bias=True, res=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.res = res
        self.bn = nn.BatchNorm1d(out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'





class GCN(nn.Module):

    def __init__(self,
                 dim_features,
                 dim_target,
                 config,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True):

        super(GCN, self).__init__()

        self.dropout = config['dropout']

        self.ingc = SAGEConv(dim_features, config['embedding_dim'])
        self.inbn = torch.nn.BatchNorm1d(config['embedding_dim'])
        self.midlayer = nn.ModuleList()

        self.bn1 = torch.nn.BatchNorm1d(config['embedding_dim'])
        self.bn2 = torch.nn.BatchNorm1d(config['embedding_dim'])
        self.bn3 = torch.nn.BatchNorm1d(config['embedding_dim'])


        for i in range(config['num_layers']):
            gcb = SAGEConv(config['embedding_dim'] , config['embedding_dim'])
            self.midlayer.append(gcb)


        self.outgc = SAGEConv(config['embedding_dim'], dim_target)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_enc = self.ingc(x, edge_index)
        x_enc = F.relu(self.bn(1,x_enc))
        x = F.dropout(x_enc, self.dropout, training=self.training)

        for i in range(len(self.midlayer)):

            midgc = self.midlayer[i]
            x = F.relu(self.bn(i+2, midgc(x, edge_index)))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.outgc(x, edge_index)

        graph_emb = global_mean_pool(x, batch)

        return graph_emb

