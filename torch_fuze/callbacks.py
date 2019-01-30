from .trainer_state import TrainerState
from .abstract_callback import AbstractCallback


class ProgressCallback(AbstractCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, state: TrainerState):
        print(f"{state.epoch}/{state.end_epoch - 1} loss = {state.run_avg_loss}, elapsed = {state.elapsed}")
