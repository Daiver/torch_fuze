import math
from bisect import bisect_right
import torch.optim as optim


class OneCycleLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_init, lr_max, lr_final, n_total_epochs, cycle_fraction=0.8, last_epoch=-1):
        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_final = lr_final
        self.n_total_epochs = n_total_epochs
        self.cycle_len = int(round(n_total_epochs * cycle_fraction))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # TODO: allow multibase
        assert len(self.base_lrs) == 1
        # TODO: ellegant expression
        if self.last_epoch <= self.cycle_len:
            prog_fraction = 2 * self.last_epoch / self.cycle_len
            if prog_fraction <= 1:
                return [self.lr_init + (self.lr_max - self.lr_init) * prog_fraction]
            else:
                return [self.lr_max + (self.lr_init - self.lr_max) * (prog_fraction - 1)]
        prog_fraction = (self.last_epoch - self.cycle_len) / (self.n_total_epochs - self.cycle_len)
        return [self.lr_init + (self.lr_final - self.lr_init) * prog_fraction]


class MultiStepLRWithLinearWarmUp(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_cutoff, init_lr_scale, milestones, gamma, last_epoch=-1):
        assert init_lr_scale < 1
        self.warmup_cutoff = warmup_cutoff
        self.init_lr_scale = init_lr_scale
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_cutoff:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]
        warmup_progress = self.last_epoch / self.warmup_cutoff
        lr_scale = (1.0 - self.init_lr_scale) * warmup_progress + self.init_lr_scale
        return [base_lr * lr_scale for base_lr in self.base_lrs]

