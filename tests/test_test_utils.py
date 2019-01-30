import unittest

from collections import OrderedDict

from .utils import is_ordered_dicts_almost_equal


class TestUtils(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
