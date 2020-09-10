import dgl
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import TAGConv


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0 and i != len(self.layers) - 1:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class TAGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(TAGCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(TAGConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(TAGConv(n_hidden, n_classes))  # activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            block = dgl.to_block(dgl.in_subgraph(g, nodes), nodes)
            input_nodes = block.srcdata[dgl.NID]

            h = x[input_nodes].to(device)
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

            y = h.cpu()
            x = y

        return y.to(device)
