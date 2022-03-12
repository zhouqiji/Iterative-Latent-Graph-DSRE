import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GCN, SGC


class GVAE(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, dropout, hops, gcn_type):
        super(GVAE, self).__init__()

        if gcn_type == 'gcn':
            self.gc_emb = GCN(input_dim, hid_dim1, hid_dim2, 1, dropout)
            self.gc_mu = GCN(hid_dim2, hid_dim2, hid_dim2, 1, dropout)
            self.gc_var = GCN(hid_dim2, hid_dim2, hid_dim2, 1, dropout)
        elif gcn_type == 'sgc':
            self.gc_emb = SGC(input_dim, hid_dim2, 1, dropout)
            self.gc_mu = SGC(hid_dim2, hid_dim2, 1, dropout)
            self.gc_var = SGC(hid_dim2, hid_dim2, 1, dropout)
        else:
            raise TypeError("The type of {} is not a valid GNN".format(gcn_type))

        self.dc = InnerProductDecoder(dropout)

    def encode(self, x, adj):
        hid1 = self.gc_emb(x, adj)
        return self.gc_mu(hid1, adj), self.gc_var(hid1, adj)

    def re_parameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, adj, node_mask):
        mu, log_var = self.encode(x, adj)
        z = self.re_parameterize(mu, log_var)
        # z = z.masked_fill_(~node_mask.bool().unsqueeze(-1), 0)
        return self.dc(z), mu, log_var


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, z.transpose(-1, -2)))
        return adj
