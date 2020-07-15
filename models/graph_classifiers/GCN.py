import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import to_dense_adj

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

class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=True,
                 res=False):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,idx=None):
        output = torch.mm(input, self.weight)

        print('out ', adj.size(), output.size())

        output = torch.spmm(adj, output)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
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

        self.ingc = Dense(dim_features, config['embedding_dim'], activation)
        self.midlayer = nn.ModuleList()
        for i in range(config['num_layers']):
            gcb = GraphConvolutionBS(config['embedding_dim'] , config['embedding_dim'], activation, withbn, withloop)
            self.midlayer.append(gcb)

        self.outgc = GraphConvolutionBS(config['embedding_dim'], dim_target)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        adj = to_dense_adj(edge_index, batch=batch)

        x_enc = self.ingc(x, adj)

        x = F.dropout(x_enc, self.dropout, training=self.training)

        for i in range(len(self.midlayer)):

            midgc = self.midlayer[i]
            x = midgc(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.outgc(x)

        graph_emb = torch.mean(x)

        return graph_emb

