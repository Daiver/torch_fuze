import unittest

from collections import OrderedDict
import torch
import torch.nn as nn

from .utils import is_ordered_dicts_almost_equal

from torch_fuze.supervised_evaluator import run_supervised_metrics


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.fc.weight.data = torch.FloatTensor([[2, 3]])
        self.fc.bias.data = torch.FloatTensor([[-1]])

    def forward(self, x):
        return self.fc(x)


class TestSupervisedEvaluator(unittest.TestCase):
    @staticmethod
    def l2_squared_loss(x, y):
        assert x.shape == y.shape
        diff = x - y
        res = (diff * diff).sum()
        return res

    def test1(self):
        model = LinearModel()
        metrics = OrderedDict([("l2", self.l2_squared_loss)])
        tensor = torch.FloatTensor
        batch = [
            (
                tensor([
                    [0, 0]
                ]),
                tensor([
                    [1]
                ])
            )
        ]

        res = run_supervised_metrics(model, metrics, batch, device="cpu")
        ans = OrderedDict([("l2", 4.0)])
        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))

    def test2(self):
        model = LinearModel()
        metrics = OrderedDict([("l2", self.l2_squared_loss)])

        tensor = torch.FloatTensor
        batch = [
            (
                tensor([
                    [0, 0],
                    [1, 3]
                ]),
                tensor([
                    [1],
                    [5]
                ])
            )
        ]

        res = run_supervised_metrics(model, metrics, batch, device="cpu")
        ans = OrderedDict([("l2", 4.0 + 25.0)])
        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))

    def test3(self):
        model = LinearModel()
        metrics = OrderedDict([("l2", self.l2_squared_loss)])

        tensor = torch.FloatTensor
        batches = [
            (
                tensor([
                    [0, 0],
                    [1, 3]
                ]),
                tensor([
                    [1],
                    [5]
                ])
            ),
            (
                tensor([
                    [1, 0]
                ]),
                tensor([
                    [-2]
                ])
            )
        ]

        res = run_supervised_metrics(model, metrics, batches, device="cpu")
        ans = OrderedDict([("l2", (29.0 + 9) / 2)])
        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))


if __name__ == '__main__':
    unittest.main()
