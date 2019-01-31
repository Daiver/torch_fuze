from .trainer_state import TrainerState
from .abstract_callback import AbstractCallback

from .supervised_evaluator import run_supervised_metrics


class ProgressCallback(AbstractCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, state: TrainerState):
        print(f"{state.epoch}/{state.end_epoch - 1} loss = {state.run_avg_loss}, elapsed = {state.elapsed}")
        for metrics_cat_name, value in state.metrics_per_category.items():
            string_to_print = " | ".join("{}: {}".format(k, v) for k, v in value.items())
            print(f"metrics_{metrics_cat_name}: {string_to_print}")


class ValidationCallback(AbstractCallback):
    def __init__(self, model, val_loader, metrics, prefix="valid", device="cuda"):
        super().__init__()
        self.model = model
        self.val_loader = val_loader
        self.metrics = metrics
        self.prefix = prefix
        self.device = device

    def on_epoch_end(self, state: TrainerState):
        metrics_vals = run_supervised_metrics(self.model, self.metrics, self.val_loader, self.device)
        state.metrics_per_category[self.prefix] = metrics_vals
