from collections import OrderedDict
import numpy as np
import torch


class SupervisedEvaluator:
    def __init__(self, model, metrics: OrderedDict, device="cuda"):
        self.model = model
        self.metrics = metrics
        self.device = device

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

    def run(self, loader):
        metrics_values_per_batch = self.mk_metrics_values(self.metrics)
        with torch.no_grad():
            model = self.model.to(self.device)
            model.eval()
            losses = []
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                for metric_name, metric_func in self.metrics.items():
                    metrics_values_per_batch[metric_name] = metric_func(outputs, targets)

        res = self.sum_metrics_values(metrics_values_per_batch, average=True)
        return res
