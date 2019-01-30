from collections import OrderedDict
import numpy as np


class SupervisedMetricsEvaluator:
    def __init__(self, metrics: OrderedDict):
        self.metrics = metrics
        self.metrics_values_per_batch = None
        self.reset()

    @staticmethod
    def mk_metrics_values(metrics: OrderedDict):
        res = OrderedDict()
        for name in metrics.keys():
            res[name] = []
        return res

    @staticmethod
    def sum_metrics_values(metrics_values: OrderedDict, average=False):
        res = OrderedDict()
        for name, values in metrics_values.items():
            res[name] = np.sum(values)
            if average and len(values) > 0:
                res[name] /= len(values)

        return res

    def reset(self):
        self.metrics_values_per_batch = self.mk_metrics_values(self.metrics)

    def process_batch(self, model_outputs, target_outputs):
        for metric_name, metric_func in self.metrics.items():
            self.metrics_values_per_batch[metric_name].append(metric_func(model_outputs, target_outputs))

    def compute_result_and_reset(self, average):
        res = self.sum_metrics_values(self.metrics_values_per_batch, average=average)
        self.reset()
        return res
