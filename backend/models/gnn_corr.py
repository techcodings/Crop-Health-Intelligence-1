import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        return F.gelu(self.lin(h))


class FieldCorrelationGNN(nn.Module):
    """
    Simple 2-layer GCN to propagate signals across fields.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.g1 = SimpleGCNLayer(in_dim, hidden_dim)
        self.g2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        h = self.g1(x, adj)
        h = self.g2(h, adj)
        return self.out(h)
