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
  model_input = Variable(input_state, volatile=True).type(FloatTensor)
  action_probabilities = model(model_input)
  action = action_probabilities.multinomial()
  return action.data


def random_use_model(configuration, steps_taken):
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
