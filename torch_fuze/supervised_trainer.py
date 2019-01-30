import time
import numpy as np

from .abstract_trainer import AbstractTrainer
from .abstract_callback import AbstractCallback


class SupervisedTrainer(AbstractTrainer):
    def __init__(self, model, criterion, device="cuda"):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

        self.state.end_epoch = 0
        self.state.run_avg_loss = None

    def run(self,
            loader,
            optimizer,
            n_epochs,
            scheduler=None,
            callbacks: list=None):
        callbacks = [] if callbacks is None else callbacks

        self.model.to(self.device)
        self.callbacks_on_training_begin(callbacks)

        self.state.end_epoch = self.state.epoch + n_epochs
        for epoch in range(self.state.epoch, self.state.end_epoch):
            self.state.epoch = epoch

            self.callbacks_on_epoch_begin(callbacks)

            losses = []
            self.model.train(True)
            start_time = time.time()
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
            end_time = time.time()
            elapsed = end_time - start_time
            self.model.train(False)

            self.state.run_avg_loss = np.mean(losses)
            print(
                f"{self.state.epoch}/{self.state.end_epoch - 1} loss = {self.state.run_avg_loss}, elapsed = {elapsed}")

            self.callbacks_on_epoch_end(callbacks)

        self.callbacks_on_training_end(callbacks)

