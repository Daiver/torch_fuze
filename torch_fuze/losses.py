import torch
import torch.nn as nn


class L12Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        assert x.shape == y.shape
        assert len(x.shape) == 3
        diff = x - y
        n_samples = x.size(0)
        n_vertices = x.size(1)
        res = torch.norm(diff, dim=-1).sum() / (n_samples * n_vertices)
        return res
