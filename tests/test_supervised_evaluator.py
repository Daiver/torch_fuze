import unittest

from collections import OrderedDict
import torch
import torch.nn as nn

from .utils import is_ordered_dicts_almost_equal

from torch_fuze.supervised_evaluator import SupervisedEvaluator


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.fc.weight.data = torch.FloatTensor([[2, 3]])
        self.fc.bias.data = torch.FloatTensor([[-1]])

    def forward(self, x):
        return self.fc(x).view(-1)


class TestSupervisedEvaluator(unittest.TestCase):
    @staticmethod
    def l2_squared_loss(x, y):
        diff = x - y
        return (diff * diff).sum()

    def test1(self):
        model = LinearModel()
        metrics = OrderedDict([("l2", self.l2_squared_loss)])
        evaluator = SupervisedEvaluator(model, metrics, device="cpu")
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

        res = evaluator.run(batch)
        ans = OrderedDict([("l2", 4.0)])
        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))


if __name__ == '__main__':
    unittest.main()
