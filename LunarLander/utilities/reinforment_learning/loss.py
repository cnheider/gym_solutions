# coding=utf-8
"""
Description: For calculating loss of the Q value function
Author: Christian Heider Nielsen
"""

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from utilities.reinforment_learning.filtering import is_nt
from utilities.reinforment_learning.replay_memory import TransitionQuadruple

def calculate_loss(current_model, target_model, transitions, configuration, use_cuda):
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
  batch = TransitionQuadruple(*zip(*transitions))
  # (S,A,S',R)^n -> (S^n,A^n,S'^n,R^n)

  states = Variable(torch.cat(batch.state))
  action_indices = Variable(torch.cat(batch.action))
  rewards = Variable(torch.cat(batch.reward))
  non_terminals = Variable(torch.cat(batch.non_terminal))
  non_terminal_futures = Variable(torch.cat(
      [future for (future, mask) in zip(batch.future, non_terminals.data) if mask]
  ))

  # Compute Q(s_t, a) - the model computes Q(s_t), then select the columns of actions taken in the batch
  Q_states = current_model(states).gather(1, action_indices)
  Q_futures = current_model(non_terminal_futures)

  if configuration.DOUBLE_DQN:
    # view ( [1,2] -> [[1],[2]] )
    Q_current_max_action_indices = Q_futures.max(1)[1].view(-1,1)
    Q_target_nt_futures = target_model(non_terminal_futures)
    Q_futures = Q_target_nt_futures.gather(1, Q_current_max_action_indices)

  # Compute V(s_{t+1})=max_{a}(Q(s_{t+1},a)), the maximum reward that can be expected in the next state s_{t+1] after taking the action a in s_{t}
    # Max expected value of all next states for terminal transition tuples, they will just be equal to 0
  V_futures = Variable(torch.zeros(configuration.BATCH_SIZE).type(FloatTensor))
  V_futures[non_terminals] = Q_futures.detach().max(1)[0] # The max value of
  # expected value of futures, detached because we should not back prop loss
  # through V_futures

  # Compute the expected Q values, max_future_state_values may be 0 resulting in only giving the reward as the expected value of taking the V(s_{t+1}) action of state s_{t+1} which may be negative
  Q_expected = rewards + (configuration.DISCOUNT_FACTOR * V_futures)

  #return F.mse_loss(Q_states, Q_expected)
  return F.smooth_l1_loss(Q_states, Q_expected)
  # Compute Huber loss, Q value difference between what the model predicts as
  # the value(how good) for taking action A in state S and what the expected
  # Q value is for what taking that action(The actual reward given by the
  # environment plus the predicted Q value of the next state)
