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

EPS = 1e-15

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

class GCN_enc(nn.Module):

    def __init__(self,
                 dim_features,
                 dim_target,
                 config,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True):

        super(GCN_enc, self).__init__()

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


        return tot

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        c_x = torch.unsqueeze(c, 1)

        c_p = c_x.expand_as(h_pl)

        batch_p, node_p, feat_p = h_pl.size()

        pos = h_pl.contiguous().view(batch_p * node_p, -1)
        sum = c_p.contiguous().view(batch_p * node_p, -1)

        sc_1 = self.f_k(pos, sum)

        c_n = c_x.expand_as(h_mi)

        batch_n, node_n, feat_n = h_mi.size()

        neg = h_mi.contiguous().view(batch_n * node_n, -1)
        sum_n = c_n.contiguous().view(batch_n * node_n, -1)

        sc_2 = self.f_k(neg, sum_n)

        return torch.sigmoid(sc_1), torch.sigmoid(sc_2)


class GCN(nn.Module):

    def __init__(self,
                 dim_features,
                 dim_target,
                 config,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True):

        super(GCN, self).__init__()

        self.encoder = GCN_enc(dim_features,
                               dim_target,
                               config,)

        self.outgc = Dense(config['embedding_dim'], dim_target, activation=F.relu)
        self.outbn = torch.nn.BatchNorm1d(dim_target)
        self.weight = nn.Linear(config['embedding_dim'], config['embedding_dim'])

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(config['embedding_dim'])



    def forward(self, data, negative_data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_n, edge_index_n, batch_n = negative_data.x, negative_data.edge_index, negative_data.batch

        pos_z = self.encoder(data)
        neg_z = self.encoder(negative_data)

        #graph
        summary = global_mean_pool(pos_z, batch)

        graph_emb = self.outgc(summary)

        pos_z, mask_ = to_dense_batch(pos_z, batch=batch)
        neg_z, mask = to_dense_batch(neg_z, batch=batch_n)

        loss_val = self.loss(pos_z, neg_z, self.sigm(summary))

        print('loss' , loss_val)

        return graph_emb, loss_val

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutal information maximization objective."""

        pos_sim, neg_sim = self.disc(summary, pos_z, neg_z)

        print(pos_sim[:5], torch.log(pos_sim[:5]), neg_sim[:5], torch.log(neg_sim[:5]))

        pos_loss = -torch.log(
            pos_sim + EPS)
        neg_loss = -torch.log(
            1 - neg_sim + EPS)

        print(pos_loss[:5], neg_loss[:5])

        return pos_loss + neg_loss

