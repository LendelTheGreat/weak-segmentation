import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 5, kernel_size=5, padding=2)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.softmax(x)
        return x
