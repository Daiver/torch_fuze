# Basically example of lr finder usage

import time
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import mlflow

import torch_fuze
from torch_fuze.utils import find_lr_supervised
from torch_fuze.lr_scheduler import OneCycleLR


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
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    # lr = 0.01
    n_epochs = 40
    batch_size = 64
    # device = "cpu"
    device = "cuda:0"

    mlflow.start_run()
    mlflow.log_param("n_epochs", n_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("device", device)

    run_start_time = mlflow.active_run().info.start_time
    readable_start_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(run_start_time / 1000))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = CIFAR10(root="data/", train=True, transform=trans, download=True)
    test_set = CIFAR10(root="data/", train=False, transform=trans, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    best_lr, summary = find_lr_supervised(model, criterion, optimizer, train_loader, 1e-9, 1, device=device)
    # torch_fuze.utils.set_lr(optimizer, best_lr * 0.1)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # plt.plot(np.log10(summary.learning_rates), summary.losses)
    # plt.plot(np.log10(summary.learning_rates), summary.smoothed_losses)
    # plt.draw()
    # plt.pause(10)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.3)
    # scheduler = OneCycleLR(optimizer, best_lr * 0.01, best_lr * 0.8, 1e-6, n_total_epochs=n_epochs, cycle_fraction=0.8)
    # scheduler = OneCycleLR(optimizer, best_lr * 0.01, best_lr * 0.5, 1e-6, n_total_epochs=n_epochs, cycle_fraction=0.8)
    # scheduler = OneCycleLR(optimizer, best_lr * 0.01, best_lr * 0.1, 1e-6, n_total_epochs=n_epochs, cycle_fraction=0.8)
    scheduler = None

    metrics = OrderedDict([
        ("loss", criterion),
        ("acc", torch_fuze.metrics.Accuracy())
    ])
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
        torch_fuze.callbacks.BestModelSaverCallback(
            model, "checkpoints/best.pt", metric_name="acc", lower_is_better=False),
        torch_fuze.callbacks.TensorBoardXCallback(f"logs/{readable_start_time}/", remove_old_logs=True),
        torch_fuze.callbacks.MLFlowCallback(
            metrics_to_track={"valid_loss", "valid_acc", "train_acc"},
            lowest_metrics_to_track={"valid_loss"},
            highest_metrics_to_track={"valid_acc"},
            files_to_save_at_every_batch={"checkpoints/best.pt"})
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, test_loader, optimizer, scheduler=scheduler, n_epochs=n_epochs, callbacks=callbacks, metrics=metrics)


if __name__ == '__main__':
    main()
