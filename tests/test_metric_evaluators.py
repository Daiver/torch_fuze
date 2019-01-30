import unittest

from collections import OrderedDict
from torch_fuze.supervised_metrics_evaluator import SupervisedMetricsEvaluator


class TestSupervisedMetricsEvaluator(unittest.TestCase):
    @staticmethod
    def l2_loss(x, y):
        return (x - y) * (x - y)

    def test1(self):

        metrics = OrderedDict([("loss", self.l2_loss)])
        evaluator = SupervisedMetricsEvaluator(metrics=metrics)

        self.assertAlmostEqual(acc, 100.0)


if __name__ == '__main__':
    unittest.main()
