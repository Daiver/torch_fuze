from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch_fuze
from torch_fuze.supervised_evaluator import run_supervised_metrics

from cifar10 import Net


def main():
    # lr = 0.01
    batch_size = 32
    # device = "cpu"
    device = "cuda"

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = CIFAR10(root="data/", train=True, transform=trans, download=True)
    test_set = CIFAR10(root="data/", train=False, transform=trans, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    model = Net()
    model.load_state_dict(torch.load("checkpoints/best.pt"))
    criterion = nn.CrossEntropyLoss()

    metrics = OrderedDict([
        ("loss", criterion),
        ("acc", torch_fuze.metrics.Accuracy())
    ])
    print(run_supervised_metrics(model, metrics, test_loader, device))


if __name__ == '__main__':
    main()
