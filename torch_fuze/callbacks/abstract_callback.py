from ..trainer_state import TrainerState


class AbstractCallback:
    def __init__(self):
        pass

    def on_training_begin(self, state: TrainerState):
        pass

    def on_training_end(self, state: TrainerState):
        pass

    def on_epoch_begin(self, state: TrainerState):
        pass

    def on_epoch_end(self, state: TrainerState):
        pass
