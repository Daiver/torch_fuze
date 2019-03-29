import matplotlib.pyplot as plt
import torch.nn as nn

import torch.optim as optim

from torch_fuze.lr_scheduler import OneCycleLR, MultiStepLRWithLinearWarmUp, CosineAnnealingLRFixedDecay


def show_1cycle():
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


def show_multistep_linear_warmup():
    model = nn.Linear(1, 2)
    optimizer = optim.SGD(model.parameters(), 1.0)

    n_epochs = 50
    scheduler = MultiStepLRWithLinearWarmUp(
        optimizer, warmup_cutoff=8, init_lr_scale=0.01, milestones=[10, 20, 30, 40], gamma=0.1)
    steps = []
    lrs = []
    for i in range(0, n_epochs):
        scheduler.step()
        steps.append(i)
        lrs.append(scheduler.get_lr()[0])
    print(lrs)

    plt.plot(steps, lrs)
    plt.show()


def show_cosine_annealing_with_fixed_decay():
    model = nn.Linear(1, 2)
    optimizer = optim.SGD(model.parameters(), 1.0)

    n_epochs = 50
    scheduler = CosineAnnealingLRFixedDecay(
        optimizer, t_max=10, eta_min=0, decay_coeff=0.5)
    steps = []
    lrs = []
    for i in range(0, n_epochs):
        scheduler.step()
        steps.append(i)
        lrs.append(scheduler.get_lr()[0])
    print(lrs)

    plt.plot(steps, lrs)
    plt.show()


if __name__ == '__main__':
    # show_multistep_linear_warmup()
    # show_1cycle()
    show_cosine_annealing_with_fixed_decay()
