from collections import OrderedDict


def ordered_dicts_almost_equal(d1: OrderedDict, d2: OrderedDict, epsilon=1e-5):
    if len(d1) != len(d2):
        return False

    for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
        if k1 != k2:
            return False
        delta = abs(v1 - v2)
        if delta > epsilon:
            return False

    return True
