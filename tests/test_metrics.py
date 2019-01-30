import unittest

import torch
from torch_fuze.metrics import Accuracy


class TestMetrics(unittest.TestCase):
    def test_accuracy1(self):
        metric = Accuracy(True)
        y_pred = torch.FloatTensor([
            [0.94, 0.3, 0.8],
            [0.0, -0.1, 0.12],
            [0.1, 0.2, -0.1055],
            [-0.01, 0.6, 0.55],
        ])

        y = torch.IntTensor([0, 2, 1, 1])
        acc = metric(y_pred, y)
        self.assertAlmostEqual(acc, 100.0)

    def test_accuracy2(self):
        metric = Accuracy(True)
        y_pred = torch.FloatTensor([
            [0.94, 0.3, 0.8],
            [0.0, 0.5, 0.12],
            [0.1, 0.2, -0.1055],
            [-0.01, 0.6, 0.55],
        ])

        y = torch.IntTensor([0, 2, 1, 1])
        acc = metric(y_pred, y)
        self.assertAlmostEqual(acc, 75.0)


if __name__ == '__main__':
    unittest.main()
