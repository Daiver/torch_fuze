from .trainer_state import TrainerState


class AbstractTrainer:
    def __init__(self):
        self.state = TrainerState()

    def run(self, *args, **kwargs):
        pass

    def callbacks_on_training_begin(self, callbacks):
        for callback in callbacks:
            callback.on_training_begin(self.state)

    def callbacks_on_training_end(self, callbacks):
        for callback in callbacks:
            callback.on_training_end(self.state)

    def callbacks_on_epoch_begin(self, callbacks):
        for callback in callbacks:
            callback.on_epoch_begin(self.state)

    def callbacks_on_epoch_end(self, callbacks):
        for callback in callbacks:
            callback.on_epoch_end(self.state)
