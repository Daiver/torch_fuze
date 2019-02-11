import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch_fuze


# Just copy-paste from https://github.com/catalyst-team/catalyst/blob/master/examples/notebook-example.ipynb
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LrFinderSummary:
    def __init__(self, losses, smoothed_losses, learning_rates, best_lr):
        self.losses = losses
        self.smoothed_losses = smoothed_losses
        self.learning_rates = learning_rates
        self.best_lr = best_lr


def get_lr(init_lr, final_lr, iteration, n_iterations):
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
        cur_lr = get_lr(init_lr, final_lr, iteration, n_iterations)
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
    return losses, smoothed_losses, lrs


def main():
    # lr = 0.01
    batch_size = 64
    # device = "cpu"
    device = "cuda"

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = CIFAR10(root="data/", train=True, transform=trans, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    losses, smoothed_losses, lrs = find_lr_supervised(
        model, criterion, optimizer, train_loader, 1e-5, 1, device=device)
    print("Lr finder finished")
    print(f"Min loss = {np.min(losses)}, best lr = {lrs[np.argmin(losses)]}")
    print(f"Min smooth loss = {np.min(smoothed_losses)}, best lr = {lrs[np.argmin(smoothed_losses)]}")

    plt.figure(1)
    plt.plot(np.log10(lrs), losses)
    # plt.figure(2)
    # plt.plot(np.log10(lrs), avg_losses)
    plt.plot(np.log10(lrs), smoothed_losses)
    plt.show()


if __name__ == '__main__':
    main()
