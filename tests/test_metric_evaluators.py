import unittest

from collections import OrderedDict
import torch

from .utils import is_ordered_dicts_almost_equal
from torch_fuze.supervised_metrics_evaluator import SupervisedMetricsEvaluator


class TestSupervisedMetricsEvaluator(unittest.TestCase):
    @staticmethod
    def l2_squared_loss(x, y):
        diff = x - y
        return (diff * diff).sum()

    @staticmethod
    def l1_loss(x, y):
        diff = x - y
        return diff.abs().sum()

    def test1(self):

        metrics = OrderedDict([("loss", self.l2_squared_loss)])
        evaluator = SupervisedMetricsEvaluator(metrics=metrics)

        output1 = torch.FloatTensor([
            [1, 2]
        ])
        target1 = torch.FloatTensor([
            [1, 2]
        ])
        evaluator.process_batch(output1, target1)

        res = evaluator.compute_result_and_reset(True)
        ans = OrderedDict([("loss", 0.0)])

        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))

    def test2(self):

        metrics = OrderedDict([("loss", self.l2_squared_loss)])
        evaluator = SupervisedMetricsEvaluator(metrics=metrics)

        output1 = torch.FloatTensor([
            [1, 1]
        ])
        target1 = torch.FloatTensor([
            [1, 2]
        ])
        evaluator.process_batch(output1, target1)

        output2 = torch.FloatTensor([
            [1, 2]
        ])
        target2 = torch.FloatTensor([
            [3, 5]
        ])
        evaluator.process_batch(output2, target2)

        res = evaluator.compute_result_and_reset(True)
        ans = OrderedDict([("loss", (1.0 + (9 + 4)) / 2)])

        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))

    def test3(self):

        metrics = OrderedDict([
            ("l2^2", self.l2_squared_loss),
            ("l1", self.l1_loss),
        ])
        evaluator = SupervisedMetricsEvaluator(metrics=metrics)

        output1 = torch.FloatTensor([
            [1, 1]
        ])
        target1 = torch.FloatTensor([
            [1, 2]
        ])
        evaluator.process_batch(output1, target1)

        output2 = torch.FloatTensor([
            [1, 2]
        ])
        target2 = torch.FloatTensor([
            [3, 5]
        ])
        evaluator.process_batch(output2, target2)

        res = evaluator.compute_result_and_reset(True)
        ans = OrderedDict([
            ("l2^2", (1.0 + (9 + 4)) / 2),
            ("l1", (1.0 + (3 + 2)) / 2),
        ])

        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))

    def test4(self):
        metrics = OrderedDict([
            ("l2^2", self.l2_squared_loss),
            ("l1", self.l1_loss),
        ])
        evaluator = SupervisedMetricsEvaluator(metrics=metrics)

        output1 = torch.FloatTensor([
            [1, 1]
        ])
        target1 = torch.FloatTensor([
            [1, 2]
        ])
        evaluator.process_batch(output1, target1)

        output2 = torch.FloatTensor([
            [1, 2]
        ])
        target2 = torch.FloatTensor([
            [3, 5]
        ])
        evaluator.process_batch(output2, target2)

        output3 = torch.FloatTensor([
            [1, 2],
            [-3, -2],
        ])
        target3 = torch.FloatTensor([
            [3, 1],
            [6, 4]
        ])
        evaluator.process_batch(output3, target3)

        res = evaluator.compute_result_and_reset(True)
        ans = OrderedDict([
            ("l2^2", (1.0 + (9 + 4) + (4 + 1 + 9**2 + 6**2)) / 3),
            ("l1", (1.0 + (3 + 2) + (2 + 1 + 9 + 6)) / 3),
        ])

        self.assertTrue(is_ordered_dicts_almost_equal(res, ans))


if __name__ == '__main__':
    unittest.main()
