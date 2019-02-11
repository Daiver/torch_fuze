import copy
from collections import OrderedDict
import torch
import torch.optim as optim


def metrics_to_nice_string(metrics: OrderedDict):
    return " | ".join("{}: {}".format(k, v) for k, v in metrics.items())


def copy_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


def set_lr(optimizer: optim.Optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
