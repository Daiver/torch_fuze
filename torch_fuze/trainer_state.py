from collections import OrderedDict


class TrainerState:
    def __init__(self, epoch=1, iteration=0):
        self.epoch = epoch
        self.end_epoch = 0
        self.iteration = iteration

        self.run_avg_loss = 1e10

        self.metrics_per_category = OrderedDict()

        self.optimizer = None
        self.scheduler = None
