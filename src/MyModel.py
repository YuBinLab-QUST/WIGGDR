import torch
from torch import nn
from torch.nn import functional as F
from .GAT import GAT


def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=adj_t.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = True, add_self_loops: bool = False,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        if self.cached:
            if not hasattr(self, "cached_adj"):
                edge_index, edge_weight = gcn_norm(
                    edge_index, self.add_self_loops)
                self.register_buffer("cached_adj", edge_index)
            edge_index = self.cached_adj
        else:
            edge_index, _ = gcn_norm(edge_index, self.add_self_loops)
        x = torch.matmul(x, self.weight)

        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class WIGGDR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, lamda=0.8, share=True, **kwargs):
        super(WIGGDR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.gat = GAT(in_channels, n_hidden=256, n_classes=out_channels, n_heads=8, dropout=0.5)
        self.gcn = GCNConv(in_channels=in_channels, out_channels=out_channels, cached=cached,
                           add_self_loops=add_self_loops, bias=bias)

        self.register_buffer("alpha", torch.tensor(lamda))
        self.register_buffer("beta", torch.tensor(1-lamda))
        self.reset_parameters()
        if share:
            self.gat.weight = self.gcn.weight

    def reset_parameters(self):
        self.gcn.reset_parameters()

    def forward(self, x, edge_index):
        adj_matrix = edge_index.to_dense()
        m = 1
        adj_matrix = torch.unsqueeze(adj_matrix, dim=2).repeat(1, 1, m)
        x1 = self.gcn(x, edge_index)
        x2=self.gat(x, adj_matrix)
        x = self.beta*F.relu(x1)+self.alpha*F.relu(x2)
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)