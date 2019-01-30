import numpy as np

from .abstract_trainer import AbstractTrainer
from .abstract_callback import AbstractCallback


class SupervisedTrainer(AbstractTrainer):
    def __init__(self, model, criterion, device="cuda"):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

        self.run_avg_loss = None

    def run(self,
            loader,
            optimizer,
            n_epochs,
            scheduler=None,
            callbacks: list[AbstractCallback]=None):
        callbacks = [] if callbacks is None else callbacks

        self.model.to(self.device)
        for callback in callbacks:
            callback.on_training_begin(self)

        for epoch in range(self.state.epoch, self.state.epoch + n_epochs):
            self.state.epoch = epoch
            for callback in callbacks:
                callback.on_epoch_begin(self)

            losses = []
            self.model.train(True)
            for iteration, (inp, target) in enumerate(loader):
                self.state.iteration = iteration

                inp, target = inp.to(self.device), target.to(self.device)
                output = self.model(inp)
                loss = self.criterion(output, target)

                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            self.model.train(False)

            self.run_avg_loss = np.mean(losses)

            for callback in callbacks:
                callback.on_epoch_end(self)

        for callback in callbacks:
            callback.on_training_end(self)

