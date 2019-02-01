from collections import OrderedDict
import time
import numpy as np

from .abstract_trainer import AbstractTrainer
from .supervised_evaluator import run_supervised_metrics
from .supervised_metrics_evaluator import SupervisedMetricsEvaluator


class SupervisedTrainer(AbstractTrainer):
    def __init__(self, model, criterion, device="cpu"):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

        self.state.end_epoch = 0
        self.state.run_avg_loss = None

    def run(self,
            train_loader,
            val_loader,
            optimizer,
            n_epochs,
            scheduler=None,
            callbacks: list=None,
            metrics: OrderedDict=None):
        callbacks = [] if callbacks is None else callbacks
        metrics = OrderedDict([("loss", self.criterion)]) if metrics is None else metrics

        self.model.to(self.device)
        self.callbacks_on_training_begin(callbacks)

        self.state.optimizer = optimizer
        self.state.scheduler = scheduler

        self.state.end_epoch = self.state.epoch + n_epochs
        for epoch in range(self.state.epoch, self.state.end_epoch):
            self.state.epoch = epoch

            self.callbacks_on_epoch_begin(callbacks)

            train_metrics_evaluator = SupervisedMetricsEvaluator(metrics)
            losses = []
            self.model.train(True)
            start_time = time.time()
            for iteration, (inp, target) in enumerate(train_loader):
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
            self.state.metrics_per_category["valid"] = run_supervised_metrics(
                self.model, metrics, val_loader, self.device)
            self.callbacks_on_epoch_end(callbacks)

        self.callbacks_on_training_end(callbacks)

