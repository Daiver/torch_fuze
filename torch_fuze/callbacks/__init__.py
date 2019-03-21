import os
import operator
import json
import torch

from ..trainer_state import TrainerState
from ..utils import metrics_to_nice_string
from ..supervised_evaluator import run_supervised_metrics
from .abstract_callback import AbstractCallback

from .tensorboardx_callback import TensorBoardXCallback
from .mlflow_callback import MLFlowCallback


class ProgressCallback(AbstractCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, state: TrainerState):
        summary_string = f"{state.epoch}/{state.end_epoch - 1} " \
                         f"loss = {state.run_avg_loss}, " \
                         f"elapsed = {state.elapsed:.2f}, "
        if state.scheduler is not None:
            summary_string += f"lr = {state.scheduler.get_lr()[0]}"
        print(summary_string)
        for metrics_cat_name, value in state.metrics_per_category.items():
            string_to_print = metrics_to_nice_string(value)
            print(f"metrics_{metrics_cat_name}: {string_to_print}")


class ValidationCallback(AbstractCallback):
    def __init__(self, model, val_loader, metrics, prefix="valid", device="cpu"):
        super().__init__()
        self.model = model
        self.val_loader = val_loader
        self.metrics = metrics
        self.prefix = prefix
        self.device = device

    def on_epoch_end(self, state: TrainerState):
        metrics_vals = run_supervised_metrics(self.model, self.metrics, self.val_loader, self.device)
        state.metrics_per_category[self.prefix] = metrics_vals


def save_model(save_path, model):
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def save_meta_info(save_path, meta_info: dict):
    with open(save_path, "w") as f:
        json.dump(meta_info, f)



class ModelSaverCallback(AbstractCallback):
    def __init__(self,
                 model,
                 save_path,
                 meta_info=None):
        super().__init__()
        self.model = model
        self.save_path = save_path
        self.meta_info = meta_info

    def save_model(self):
        save_model(self.save_path, self.model)
        if self.meta_info is not None:
            meta_info_path = self.save_path + ".json"
            save_meta_info(meta_info_path, self.meta_info)

    def on_epoch_end(self, state: TrainerState):
        self.save_model()


class BestModelSaverCallback(ModelSaverCallback):
    def __init__(self,
                 model,
                 save_path,
                 metric_name="loss",
                 metric_category="valid",
                 lower_is_better=True,
                 meta_info=None,
                 verbose=False):
        super().__init__(model, save_path, meta_info)

        self.metric_name = metric_name
        self.metric_category = metric_category
        self.best_value = None
        self.comparison_operator = operator.lt if lower_is_better else operator.gt
        self.verbose = verbose


    def on_epoch_end(self, state: TrainerState):
        is_first_epoch = self.best_value is None
        current_metric_val = state.metrics_per_category[self.metric_category][self.metric_name]
        is_current_val_better = is_first_epoch or (self.comparison_operator(current_metric_val, self.best_value))
        if is_current_val_better:
            self.best_value = current_metric_val
            if self.verbose:
                print(f"New best value: {self.best_value}. Saving")
            self.save_model()

