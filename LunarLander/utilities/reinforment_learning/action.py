import math
import random

import torch
from configs.default_config import *
from neodroid.models import Reaction, Motion
from torch.autograd import Variable

_use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor


def sample_action(input_state, model, steps_done):
  sample = random.random()
  eps_threshold = EPS_END + ((EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
  if sample > eps_threshold:
    vari = Variable(input_state, volatile=True).type(FloatTensor)
    action_probs = model(vari)
    action = action_probs.multinomial()
    #maximum_reward_action = action.data.max(1)
    #maximum_reward_action_index = maximum_reward_action[1]
    #return maximum_reward_action_index.view(1, 1)
    return action.data
  else:
    return LongTensor(
        [[random.randrange(ARCHITECTURE_CONFIGURATION['output_size'])]])

def testing_sample_action(input_state, model):
  vari = Variable(input_state, volatile=True).type(FloatTensor)
  action_probablities = model(vari)
  action = action_probablities.multinomial()
  return action.data
