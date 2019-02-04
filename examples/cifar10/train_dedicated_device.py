from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch_fuze
import pretrainedmodels


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


class Net2(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = pretrainedmodels.resnet50()
        self.add_module('backbone', self.backbone)
        self.fc_final = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.backbone.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, 2048)
        x = self.fc_final(x)
        return x


def main():
    print(f"Torch version: {torch.__version__}, CUDA: {torch.version.cuda}, Fuze version: {torch_fuze.__version__}")

    # lr = 0.01
    # batch_size = 32
    # batch_size = 64
    batch_size = 128
    # device = "cpu"
    device = "cuda:1"

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU name: {torch.cuda.get_device_name(int(device.split(':')[-1]))}")

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not exist, download mnist dataset
    train_set = CIFAR10(root="data/", train=True, transform=trans, download=True)
    test_set = CIFAR10(root="data/", train=False, transform=trans, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    # model = Net()
    model = Net2()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.3)

    metrics = OrderedDict([
        ("loss", criterion),
        ("acc", torch_fuze.metrics.Accuracy())
    ])
    callbacks = [
        torch_fuze.callbacks.ProgressCallback(),
        # torch_fuze.callbacks.BestModelSaverCallback(
        #     model, "checkpoints/best.pt", metric_name="acc", lower_is_better=False),
        # torch_fuze.callbacks.TensorBoardXCallback("logs", remove_old_logs=True),
        # torch_fuze.callbacks.MLFlowCallback(
        #     metrics_to_track={"valid_loss", "valid_acc", "train_acc"},
        #     lowest_metrics_to_track={"valid_loss"},
        #     highest_metrics_to_track={"valid_acc"},
        #     files_to_save_at_every_batch={"checkpoints/best.pt"})
    ]
    trainer = torch_fuze.SupervisedTrainer(model, criterion, device)
    trainer.run(
        train_loader, test_loader, optimizer, scheduler=scheduler, n_epochs=200, callbacks=callbacks, metrics=metrics)


if __name__ == '__main__':
    main()
