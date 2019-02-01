from ..trainer_state import TrainerState
from .abstract_callback import AbstractCallback

import os
import shutil
import tensorboardX


class TensorBoardXCallback(AbstractCallback):
    def __init__(self, log_dir=None, remove_old_logs=False):
        super().__init__()
        if remove_old_logs:
            shutil.rmtree(log_dir, ignore_errors=True)
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, state: TrainerState):
        global_step = state.epoch

        for cat_name, metrics in state.metrics_per_category.items():
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar("/".join((cat_name, metric_name)), metric_value, global_step=global_step)
