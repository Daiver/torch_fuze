import math
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch.nn as nn
    model = nn.Linear(1, 2)
    optimizer = optim.SGD(model.parameters(), 1.0)

    n_epochs = 50
    scheduler = OneCycleLR(optimizer, lr_init=0.1, lr_max=2, lr_final=1e-2, n_total_epochs=n_epochs, cycle_fraction=0.8)
    steps = []
    lrs = []
    for i in range(0, n_epochs):
        scheduler.step()
        steps.append(i)
        lrs.append(scheduler.get_lr()[0])

    plt.plot(steps, lrs)
    plt.show()

