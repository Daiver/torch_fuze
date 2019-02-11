import unittest

from collections import OrderedDict
import torch

from .utils import is_ordered_dicts_almost_equal, is_tensors_almost_equal


class TestTestUtils(unittest.TestCase):
    def test_ordered_dicts_almost_equal1(self):
        d1 = OrderedDict([('1', 2.0)])
        d2 = OrderedDict([('1', 2.0)])
        self.assertTrue(is_ordered_dicts_almost_equal(d1, d2))

    def test_ordered_dicts_almost_equal2(self):
        d1 = OrderedDict([('1', 2.0), ('5', 3)])
        d2 = OrderedDict([('1', 2.0)])
        self.assertFalse(is_ordered_dicts_almost_equal(d1, d2))

    def test_ordered_dicts_almost_equal3(self):
        d1 = OrderedDict([('1', 2.0), ('5', 3)])
        d2 = OrderedDict([('1', 2.0), ('5', 3.1)])
        self.assertFalse(is_ordered_dicts_almost_equal(d1, d2))

    def test_ordered_dicts_almost_equal4(self):
        d1 = OrderedDict([('1', 2.0), ('5', 3)])
        d2 = OrderedDict([('1', 2.0), ('5', 3.01)])
        self.assertTrue(is_ordered_dicts_almost_equal(d1, d2, epsilon=0.1))

    def test_ordered_dicts_almost_equal5(self):
        d1 = OrderedDict([('1', 2.0), ('5', 3)])
        d2 = OrderedDict([('1', 2.0), ('5', 3.0)])
        self.assertTrue(is_ordered_dicts_almost_equal(d1, d2))

    def test_ordered_dicts_almost_equal6(self):
        d1 = OrderedDict([('1', 2.0), ('5', 3.0)])
        d2 = OrderedDict([('5', 3.0), ('1', 2.0)])
        self.assertFalse(is_ordered_dicts_almost_equal(d1, d2))

    def test_is_tensors_almost_equal01(self):
        t1 = torch.FloatTensor([1, 2, 3])
        t2 = torch.FloatTensor([1, 2, 3])
        self.assertTrue(is_tensors_almost_equal(t1, t2))

    def test_is_tensors_almost_equal02(self):
        t1 = torch.FloatTensor([1, 2, 3])
        t2 = torch.FloatTensor([1, 5, 3])
        self.assertFalse(is_tensors_almost_equal(t1, t2))

    def test_is_tensors_almost_equal03(self):
        t1 = torch.FloatTensor([[1, 0], [0, 1]])
        t2 = torch.FloatTensor([[1, 0], [0, 1.01]])
        self.assertTrue(is_tensors_almost_equal(t1, t2, epsilon=0.1))

    def test_is_tensors_almost_equal04(self):
        t1 = torch.FloatTensor([[1, 0], [0, 1]])
        t2 = torch.FloatTensor([[1, 0], [0, 1.01]])
        self.assertFalse(is_tensors_almost_equal(t1, t2))


if __name__ == '__main__':
    unittest.main()
