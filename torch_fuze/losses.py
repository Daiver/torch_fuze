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


class WingLoss(nn.Module):
    def __init__(self, l1_log_cutoff, epsilon):
        super().__init__()
        self.l1_log_cutoff = l1_log_cutoff
        self.epsilon = epsilon

        log_val = torch.log(torch.FloatTensor([1 + self.l1_log_cutoff / self.epsilon])).item()
        self.link_constant = self.l1_log_cutoff - self.l1_log_cutoff * log_val

    def forward(self, x, y):
        assert x.shape == y.shape
        n_dims = len(x.shape)
        n_samples = x.size(0) if n_dims > 0 else 1
        n_vertices = x.size(1) if n_dims > 1 else 1
        diff = x - y

        abs_diff = diff.abs()
        is_item_in_l1_zone = torch.ge(abs_diff, self.l1_log_cutoff).float()
        is_item_in_log_zone = 1 - is_item_in_l1_zone

        log_val = self.l1_log_cutoff * torch.log(1 + abs_diff / self.epsilon)
        res = is_item_in_l1_zone * (abs_diff - self.link_constant) + is_item_in_log_zone * log_val
        res = res.sum() / (n_samples * n_vertices)
        return res
