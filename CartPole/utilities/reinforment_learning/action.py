# coding=utf-8
import math
import random

import torch
from torch.autograd import Variable

_use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor
ActionTensorType = LongTensor


def sample_action(environment,
                  model,
                  state,
                  configuration,
                  total_steps_taken=0):
  """

  :param input_state:
  :param model:
  :return:
  """
  if epsilon_random(configuration, total_steps_taken) and \
          total_steps_taken > configuration.INITIAL_OBSERVATION_PERIOD:
    model_input = Variable(state, volatile=True).type(FloatTensor)
    action_probabilities = model(model_input)
    return action_probabilities.data.max(1)[1].view(1, 1)
  else:
    environment_action = environment.action_space.sample()
    return ActionTensorType([[environment_action]])


def epsilon_random(configuration, steps_taken):
  """

  :param configuration:
  :param steps_taken:
  :return:
  """
  if steps_taken == 0:
    return True
  sample = random.random()
  eps_threshold = configuration.EPS_END + (
    (configuration.EPS_START - configuration.EPS_END) * math.exp(
        -1. * steps_taken / configuration.EPS_DECAY))
  return sample > eps_threshold
