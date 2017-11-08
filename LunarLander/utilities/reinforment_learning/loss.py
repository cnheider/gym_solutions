# coding=utf-8
"""
Description: For calculating loss of the Q value function
Author: Christian Heider Nielsen
"""

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from utilities.reinforment_learning.filtering import is_non_terminal
from utilities.reinforment_learning.replay_memory import TransitionQuadruple

def calculate_loss(model, target_model, transitions, configuration, use_cuda):
  """

  :param use_cuda:
  :param model:
  :param target_model:
  :param transitions:
  :param configuration:
  :return:
  """

  FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
  ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

  batch = TransitionQuadruple(*zip(*transitions))
  # Inverse of zip, transpose the batch, http://stackoverflow.com/a/19343/3343043

  non_terminal_mask = ByteTensor(
      tuple(
          map(is_non_terminal, batch.next_state)
      ))
  # Compute a indexing mask of non-final states and concatenate the batch elements

  non_terminal_next_states = [s for s in batch.next_state if s is not None]
  non_terminal_next_states = Variable(torch.cat(non_terminal_next_states),
                                      volatile=True)
  # (Volatile) don not back propagate error through the expected action values

  states = Variable(torch.cat(batch.state))
  action_indices = Variable(torch.cat(batch.action))
  rewards_given_for_action = Variable(torch.cat(batch.reward))

  q_of_actions_taken = model(states).gather(1, action_indices)
  # Compute Q(s_t, a) - the model computes Q(s_t), then select the columns of actions taken in the batch

  zeroes = torch.zeros(configuration.BATCH_SIZE).type(FloatTensor)
  v_next_states = Variable(zeroes)
  q_next_states = target_model(non_terminal_next_states)
  # Use the target network for estimating the Q value of next states, stabilising training
  v_next_states[non_terminal_mask] = q_next_states.max(1)[0]
  # Compute V(s_{t+1})=max_{a}(Q(s_{t+1},a)), the maximum reward that can be expected in the next state s_{t+1] after taking the action a in s_{t}

  v_next_states.volatile = False  # Do not mess up the loss with volatile
  # flag
  q_expected = rewards_given_for_action + (configuration.GAMMA * v_next_states)
  # Compute the expected Q values, max_future_state_values may be 0 resulting in only giving the reward as the expected value of taking the V(s_{t+1}) action of state s_{t+1} which may be negative

  return F.smooth_l1_loss(q_of_actions_taken, q_expected)
  # Compute Huber loss, Q value difference between what the model predicts as the value(how good) for taking action A in state S and what the expected Q value is for what taking that action(The actual reward given by plus environment the Q value of the next state)
