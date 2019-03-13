import copy
import random
from collections import OrderedDict
import torch
import torch.optim as optim
import numpy as np


def metrics_to_nice_string(metrics: OrderedDict):
    return " | ".join("{}: {}".format(k, v) for k, v in metrics.items())


def copy_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


def set_lr(optimizer: optim.Optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def manual_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
