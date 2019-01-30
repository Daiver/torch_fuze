from .abstract_trainer import AbstractTrainer


class AbstractCallback:
    def __init__(self):
        pass

    def on_training_begin(self, trainer: AbstractTrainer):
        pass

    def on_training_end(self, trainer: AbstractTrainer):
        pass

    def on_epoch_begin(self, trainer: AbstractTrainer):
        pass

    def on_epoch_end(self, trainer: AbstractTrainer):
        pass
