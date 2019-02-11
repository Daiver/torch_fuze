import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from torch_fuze.utils import find_lr_supervised


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
    best_lr, summary = find_lr_supervised(model, criterion, optimizer, train_loader, 1e-5, 1, device=device)
    print("Lr finder finished")
    print(f"Best lr = {best_lr}")
    print(f"Min loss = {np.min(summary.losses)}, best lr = {summary.learning_rates[np.argmin(summary.losses)]}")
    print(f"Min smooth loss = {np.min(summary.smoothed_losses)}, "
          f"best lr = {summary.learning_rates[np.argmin(summary.smoothed_losses)]}")

    plt.figure(1)
    plt.plot(np.log10(summary.learning_rates), summary.losses)
    # plt.figure(2)
    # plt.plot(np.log10(lrs), avg_losses)
    plt.plot(np.log10(summary.learning_rates), summary.smoothed_losses)
    plt.show()


if __name__ == '__main__':
    main()
