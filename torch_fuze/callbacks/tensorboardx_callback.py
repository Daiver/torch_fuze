from ..trainer_state import TrainerState
from .abstract_callback import AbstractCallback

import tensorboardX


class TensorBoardXCallback(AbstractCallback):
    def __init__(self, log_dir=None):
        super().__init__()
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, state: TrainerState):
        global_step = state.epoch

        for cat_name, metrics in state.metrics_per_category.items():
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar("/".join((cat_name, metric_name)), metric_value, global_step=global_step)
