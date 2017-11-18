# coding=utf-8
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
  def __init__(self, configuration):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(configuration['input_channels'], 32, kernel_size=8,
                           stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.fc1 = nn.Linear(7 * 7 * 64, 512)
    self.fc2 = nn.Linear(512, configuration['output_size'])

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(x.size(0), -1) # Flatten
    x = F.relu(self.fc1(x))
    return self.fc2(x)