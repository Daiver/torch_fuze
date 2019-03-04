from collections import OrderedDict
import torch

from .supervised_metrics_evaluator import SupervisedMetricsEvaluator


def run_supervised_metrics(model, metrics, loader, device):
    metrics_evaluator = SupervisedMetricsEvaluator(metrics)
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            metrics_evaluator.process_batch(outputs, targets)

    # print([float(x.detach().cpu().numpy()) for x in metrics_evaluator.metrics_values_per_batch["loss"]])
    res = metrics_evaluator.compute_result_and_reset(True)
    return res
