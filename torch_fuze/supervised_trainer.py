from .abstract_trainer import AbstractTrainer
from .abstract_callback import AbstractCallback


class SupervisedTrainer(AbstractTrainer):
    def __init__(self, model, criterion, device="cuda"):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

    def run(self,
            loader,
            optimizer,
            start_epoch,
            end_epoch,
            callbacks: list[AbstractCallback]=None):
        callbacks = [] if callbacks is None else callbacks

        self.model.to(self.device)

        for callback in callbacks:
            callback.on_training_begin(self)



        for callback in callbacks:
            callback.on_training_end(self)
