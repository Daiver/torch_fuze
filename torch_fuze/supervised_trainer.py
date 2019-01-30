from collections import OrderedDict
import time
import numpy as np

from .abstract_trainer import AbstractTrainer
from .abstract_callback import AbstractCallback
from .supervised_metrics_evaluator import SupervisedMetricsEvaluator


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
            callbacks: list=None,
            train_metrics: OrderedDict=None):
        callbacks = [] if callbacks is None else callbacks
        train_metrics = OrderedDict([("loss", self.criterion)]) if train_metrics is None else train_metrics

        self.model.to(self.device)
        self.callbacks_on_training_begin(callbacks)

        self.state.end_epoch = self.state.epoch + n_epochs
        for epoch in range(self.state.epoch, self.state.end_epoch):
            self.state.epoch = epoch

            self.callbacks_on_epoch_begin(callbacks)

            train_metrics_evaluator = SupervisedMetricsEvaluator(train_metrics)
            losses = []
            self.model.train(True)
            start_time = time.time()
            for iteration, (inp, target) in enumerate(loader):
                self.state.iteration = iteration

                inp, target = inp.to(self.device), target.to(self.device)
                output = self.model(inp)
                train_metrics_evaluator.process_batch(output, target)
                loss = self.criterion(output, target)

                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            end_time = time.time()
            self.state.elapsed = end_time - start_time
            self.model.train(False)

            self.state.run_avg_loss = np.mean(losses)

            self.state.metrics_per_category["train"] = train_metrics_evaluator.compute_result_and_reset(True)
            self.callbacks_on_epoch_end(callbacks)

        self.callbacks_on_training_end(callbacks)

