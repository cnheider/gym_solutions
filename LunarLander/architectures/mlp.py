# coding=utf-8
"""
Description: DQN
Author: Christian Heider Nielsen
"""
from torch import nn
from torch.nn import functional as F
import numpy as np

class LinearOutputAffineMLP(nn.Module):
  """
  OOOO input_size
  |XX|                                        fc1
  OOOO hidden_layer_size * (Weights,Biases)
  |XX|                                        fc2
  OOOO hidden_layer_size * (Weights,Biases)
  |XX|                                        fc3
  0000 output_size * (Weights,Biases)
  """

  def __init__(self, configuration):
    super(LinearOutputAffineMLP, self).__init__()
    self.fc1 = nn.Linear(configuration['input_size'],
                         configuration['hidden_layers'][0])
    self.fc2 = nn.Linear(configuration['hidden_layers'][0],
                         configuration['hidden_layers'][1])
    self.fc3 = nn.Linear(configuration['hidden_layers'][1],
                         configuration['hidden_layers'][2])
    self.fc4 = nn.Linear(configuration['hidden_layers'][2],
                         configuration['output_size'])

  def forward(self, x):
    """

    :param x:
    :return x:
    """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return self.fc4(x)