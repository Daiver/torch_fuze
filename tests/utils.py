from collections import OrderedDict
import torch


def is_ordered_dicts_almost_equal(d1: OrderedDict, d2: OrderedDict, epsilon=1e-5):
    if len(d1) != len(d2):
        return False

    for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
        if k1 != k2:
            return False
        delta = abs(v1 - v2)
        if delta > epsilon:
            return False

    return True


def is_tensors_almost_equal(t1, t2, epsilon=1e-9):
    return torch.dist(t1, t2).item() < epsilon
