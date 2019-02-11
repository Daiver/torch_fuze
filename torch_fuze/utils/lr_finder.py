import numpy as np


class LrFinderSummary:
    def __init__(self, losses, smoothed_losses, learning_rates, best_lr):
        self.losses = losses
        self.smoothed_losses = smoothed_losses
        self.learning_rates = learning_rates
        self.best_lr = best_lr


def get_logscale_lr(init_lr, final_lr, iteration, n_iterations):
    return init_lr * (final_lr / init_lr) ** (iteration / n_iterations)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def find_lr_supervised(
        model,
        criterion,
        optimizer,
        loader,
        init_lr,
        final_lr,
        device="cpu",
        avg_loss_momentum=0.98):
    n_items = len(loader)

    n_iterations = n_items - 1

    model = model.to(device)
    lrs = []
    losses = []
    avg_losses = []
    smoothed_losses = []
    avg_loss = 0

    for iteration, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        cur_lr = get_logscale_lr(init_lr, final_lr, iteration, n_iterations)
        set_lr(optimizer, cur_lr)
        lrs.append(cur_lr)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())

        avg_loss = (avg_loss_momentum * avg_loss + (1 - avg_loss_momentum) * losses[-1])
        avg_losses.append(avg_loss)
        smoothed_losses.append(avg_losses[-1] / (1 - avg_loss_momentum**(iteration+1)))

        loss.backward()
        optimizer.step()

    best_lr = lrs[np.argmin(smoothed_losses)]
    return best_lr, LrFinderSummary(losses=losses, smoothed_losses=smoothed_losses, learning_rates=lrs, best_lr=best_lr)
