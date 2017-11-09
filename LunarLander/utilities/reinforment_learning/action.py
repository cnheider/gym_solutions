# coding=utf-8
import math
import random

import torch
from torch.autograd import Variable

_use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor


def sample_action(input_state, model):
  """

  :param input_state:
  :param model:
  :return:
  """
  model_input = Variable(input_state, volatile=True).type(FloatTensor) #
  # Volatile because the loss computation step will recalculate model
  # predictions for the gradient update
  action_probabilities = model(model_input)
  #action = action_probabilities.multinomial()
  action = action_probabilities.data.max(1)[1].view(1, 1)
  #print(model_input,action_probabilities,action)
  return action


def maybe_sample_from_model(configuration, steps_taken):
  """

  :param configuration:
  :param steps_taken:
  :return:
  """
  sample = random.random()
  eps_threshold = configuration.EPS_END + (
    (configuration.EPS_START - configuration.EPS_END) * math.exp(
        -1. * steps_taken / configuration.EPS_DECAY))
  return sample > eps_threshold
