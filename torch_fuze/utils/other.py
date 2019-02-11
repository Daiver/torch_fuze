import copy
from collections import OrderedDict
import torch


def metrics_to_nice_string(metrics: OrderedDict):
    return " | ".join("{}: {}".format(k, v) for k, v in metrics.items())


def copy_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)
