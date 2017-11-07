from torch import nn
from torch.nn import functional as F


class MyDQN(nn.Module):
  def __init__(self, configuration):
    super(MyDQN, self).__init__()
    self.fc1 = nn.Linear(configuration['input_size'], configuration['hidden_layers_size'])
    self.fc2 = nn.Linear(configuration['hidden_layers_size'], configuration['output_size'])

  def forward(self, x):
    x = F.relu(self.fc1(x))
    action_scores = self.fc2(x)
    return F.softmax(action_scores)
