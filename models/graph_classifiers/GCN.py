import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import add_self_loops, to_dense_batch, to_dense_adj
from torch_geometric.nn import SAGEConv,GraphConv, global_mean_pool

from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

def loss_function(mu, logvar, decoded, real, n_nodes = 8):

    print('values ', mu.size(), logvar.size(), decoded.size(), real.size())

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return KLD

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

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1,2)))
        return adj

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

        self.ingc = GraphConv(dim_features, config['embedding_dim'])
        self.inbn = torch.nn.BatchNorm1d(config['embedding_dim'])
        self.midlayer = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(config['num_layers']):
            gcb = GraphConv(config['embedding_dim'] , config['embedding_dim'])
            self.midlayer.append(gcb)
            bn2 = torch.nn.BatchNorm1d(config['embedding_dim'])
            self.bns.append(bn2)


        self.outgc = Dense(config['embedding_dim'], dim_target, activation=F.relu)
        self.outbn = torch.nn.BatchNorm1d(dim_target)



    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_enc = self.ingc(x, edge_index)
        x_enc = F.relu(self.inbn(x_enc))
        x = F.dropout(x_enc, self.dropout, training=self.training)
        tot = x

        for i in range(len(self.midlayer)):

            midgc = self.midlayer[i]
            bn = self.bns[i]

            x = F.relu(bn(midgc(x, edge_index)))
            x = F.dropout(x, self.dropout, training=self.training)

            tot = tot + x


        #graph
        graph_emb = global_mean_pool(tot, batch)

        graph_emb = self.outgc(graph_emb)

        return graph_emb

