from collections import OrderedDict


def metrics_to_nice_string(metrics: OrderedDict):
    return " | ".join("{}: {}".format(k, v) for k, v in metrics.items())
