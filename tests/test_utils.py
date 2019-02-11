import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader

from torch_fuze.utils import copy_model

from .utils import is_tensors_almost_equal


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


class TestUtils(unittest.TestCase):
    def test_copy_model01(self):
        model = SimpleModel()
        model2 = copy_model(model)

        self.assertTrue(torch.eq(model.fc.weight.data, model2.fc.weight.data))
        self.assertTrue(torch.eq(model.fc.bias.data, model2.fc.bias.data))

    def test_copy_model02(self):
        model = SimpleModel()

        model.fc.weight.data = torch.FloatTensor([[1]])
        model.fc.bias.data = torch.FloatTensor([-10])

        model2 = copy_model(model)

        model2.fc.weight.data = torch.FloatTensor([[6]])

        self.assertFalse(is_tensors_almost_equal(model.fc.weight.data, model2.fc.weight.data))
        self.assertTrue(is_tensors_almost_equal(model.fc.bias.data, model2.fc.bias.data))

    def test_copy_model03(self):
        model = SimpleModel()

        model.fc.weight.data = torch.FloatTensor([[1]])
        model.fc.bias.data = torch.FloatTensor([-10])

        model2 = copy_model(model)

        batch_x = torch.FloatTensor([
            [12], [-2]
        ])
        batch_y = torch.FloatTensor([
            [1], [2]
        ])

        optimizer = optim.SGD(params=model2.parameters(), lr=1e-1)
        y_pred = model2(batch_x)
        loss = F.mse_loss(y_pred, batch_y)
        loss.backward()
        optimizer.step()

        self.assertFalse(is_tensors_almost_equal(model.fc.weight.data, model2.fc.weight.data))
        self.assertFalse(is_tensors_almost_equal(model.fc.bias.data, model2.fc.bias.data))

        self.assertTrue(is_tensors_almost_equal(model.fc.weight.data, torch.FloatTensor([[1]])))
        self.assertTrue(is_tensors_almost_equal(model.fc.bias.data, torch.FloatTensor([-10])))


if __name__ == '__main__':
    unittest.main()
