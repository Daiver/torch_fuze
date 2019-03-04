import unittest

import torch
from torch_fuze.losses import L12Loss


class TestLosses(unittest.TestCase):
    def test_l12_01(self):
        metric = L12Loss()
        x = torch.FloatTensor([[
            [1, 2, 3]
        ]])
        y = torch.FloatTensor([[
            [1, 2, 3]
        ]])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), 0)

    def test_l12_02(self):
        metric = L12Loss()
        x = torch.FloatTensor([[
            [0, 0, 2]
        ]])
        y = torch.FloatTensor([[
            [0, 0, 0]
        ]])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), 2)

    def test_l12_03(self):
        metric = L12Loss()
        x = torch.FloatTensor([[
            [0, 0, 0]
        ]])
        y = torch.FloatTensor([[
            [0, 4, 3]
        ]])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), 5)

    def test_l12_04(self):
        metric = L12Loss()
        x = torch.FloatTensor([[
            [0, 4, 0],
            [0, 9, -1]
        ]])
        y = torch.FloatTensor([[
            [0, 4, 3],
            [0, 5, 2],
        ]])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), (3 + 5) / 2)

    def test_l12_05(self):
        metric = L12Loss()
        x = torch.FloatTensor([
            [[0, 4, 0]],
            [[0, 9, -1]]
        ])
        y = torch.FloatTensor([
            [[0, 4, 3]],
            [[0, 5, 2]],
        ])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), (3 + 5) / 2)

    def test_l12_06(self):
        metric = L12Loss()
        x = torch.FloatTensor([[[5, 0, 1]]])
        y = torch.FloatTensor([[[6, 2, 3]]])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), 3)

    def test_l12_07(self):
        metric = L12Loss()
        x = torch.FloatTensor([
            [
                [0, 4, 0],
                [1, 2, 3],
            ],
            [
                [0, 9, -1],
                [5, 0, 1],
            ]
        ])
        y = torch.FloatTensor([
            [
                [0, 4, 3],
                [1, 2, 3],
            ],
            [
                [0, 5, 2],
                [6, 2, 3],
            ],
        ])

        loss = metric(x, y)
        self.assertAlmostEqual(loss.item(), (3 + 5 + 0 + 3) / 4)
