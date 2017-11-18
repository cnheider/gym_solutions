# coding=utf-8
"""
Description: For calculating loss of the Q value function
Author: Christian Heider Nielsen
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from utilities.reinforment_learning.replay_memory import Transition

def optimise_model(model,
                   target_model,
                   optimiser,
                   transitions,
                   configuration,
                   use_cuda):
  """

  :param use_cuda:
  :param current_model:
  :param target_model:
  :param transitions:
  :param configuration:
  :return:
  """

  FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

  # Inverse of zip, transpose the batch, http://stackoverflow.com/a/19343/3343043
  batch = Transition(*zip(*transitions))
  # (S,A,S',R)^n -> (S^n,A^n,S'^n,R^n)

  states = Variable(torch.cat(batch.state))
  action_indices = Variable(torch.cat(batch.action))
  rewards = Variable(torch.cat(batch.reward))
  non_terminals = Variable(torch.cat(batch.non_terminal))
  non_terminal_successor_states = Variable(torch.cat(
      [state for (state, non_terminal) in zip(batch.successor_states,
                                                non_terminals.data)
       if non_terminal]
  ))

  Q_states = model(states).gather(1, action_indices)
  Q_successors = model(non_terminal_successor_states)

  if configuration.DOUBLE_DQN:
    Q_successors = target_model(non_terminal_successor_states)

  V_successors = Variable(torch.zeros(configuration.BATCH_SIZE).type(FloatTensor))
  V_successors[non_terminals] = Q_successors.detach().max(1)[0]

  Q_expected = rewards + (configuration.DISCOUNT_FACTOR * V_successors)

  loss_variable = F.smooth_l1_loss(Q_states, Q_expected)

  optimiser.zero_grad()
  loss_variable.backward()
  for params in model.parameters():
    params.grad.data.clamp_(-1, 1)
  optimiser.step()

  return loss_variable.data.cpu().numpy().mean()