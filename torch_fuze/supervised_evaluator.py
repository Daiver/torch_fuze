from collections import OrderedDict
import torch

from .supervised_metrics_evaluator import SupervisedMetricsEvaluator


class SupervisedEvaluator:
    def __init__(self, model, metrics: OrderedDict, device="cuda"):
        self.model = model
        self.metrics = metrics
        self.device = device

    def run(self, loader):
        metrics_evaluator = SupervisedMetricsEvaluator(self.metrics)
        with torch.no_grad():
            model = self.model.to(self.device)
            model.eval()
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                metrics_evaluator.process_batch(outputs, targets)

        res = metrics_evaluator.compute_result_and_reset(True)
        return res
