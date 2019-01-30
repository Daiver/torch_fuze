import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self, average=True):
        assert average
        super().__init__()
        self.average = True

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        assert y_pred.dim() == 2
        assert y.dim() == 1
        batch_size = y.size(0)
        assert batch_size == y_pred.size(0)

        y_pred = y_pred.detach()
        y = y.detach()

        y_pred = torch.argmax(y_pred, dim=1).int()
        y = y.int()

        acc = torch.eq(y_pred, y).sum().item()
        acc /= batch_size
        acc *= 100

        return acc
