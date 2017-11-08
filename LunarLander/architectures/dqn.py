from torch import nn
from torch.nn import functional as F


class MyDQN(nn.Module):
  def __init__(self, configuration):
    super(MyDQN, self).__init__()
    self.fc1 = nn.Linear(configuration['input_size'], configuration['hidden_layers_size'])
    self.fc2 = nn.Linear(configuration['hidden_layers_size'], configuration['hidden_layers_size'])
    self.fc3 = nn.Linear(configuration['hidden_layers_size'], configuration['output_size'])

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.softmax(x)
