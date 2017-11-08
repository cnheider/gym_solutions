import glob
import datetime
import time
from itertools import count
import os

import os
import torch
import torch.optim as optimisation
import neodroid as neo
import torch.nn.functional as F
import gym

from PIL import ImageFile
from torch.autograd import Variable
from visdom import Visdom

from utilities.data.processing import data_transform
from utilities.visualisation import update_visualiser

from utilities.reinforment_learning.action import sample_action
from utilities.reinforment_learning.filtering import is_non_terminal
from utilities.reinforment_learning.replay_memory import (
  ReplayMemory,
  TransitionQuadruple)

import configs.default_config as configuration

ImageFile.LOAD_TRUNCATED_IMAGES = True

_use_cuda = False
if configuration.USE_CUDA_IF_AVAILABLE:
  _use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor

def calculate_loss(model, target_model, transitions, configuration):
  batch = TransitionQuadruple(*zip(*transitions)) # Inverse of zip, transpose the batch, http://stackoverflow.com/a/19343/3343043

  non_terminal_mask = ByteTensor(
      tuple(
          map(is_non_terminal, batch.next_state)
      ))  # Compute a indexing mask of non-final states and concatenate the batch elements

  non_terminal_next_states = [s for s in batch.next_state if s is not None]
  non_terminal_next_states = Variable(torch.cat(non_terminal_next_states), volatile=True) # (Volatile) dont backprop through the expected action values

  states = Variable(torch.cat(batch.state))
  action_indices = Variable(torch.cat(batch.action))
  rewards_given_for_action = Variable(torch.cat(batch.reward))

  q_of_actions_taken = model(states).gather(1, action_indices) # Compute Q(s_t, a) - the model computes Q(s_t), then select the columns of actions taken in the batch

  zeroes = torch.zeros(configuration.BATCH_SIZE).type(FloatTensor)
  v_next_states = Variable(zeroes)
  q_next_states = target_model(non_terminal_next_states) # Use the target network for estimating the Q value of next states, stabilising training
  v_next_states[non_terminal_mask] = q_next_states.max(1)[0]   # Compute V(s_{t+1})=max_{a}(Q(s_{t+1},a)), the maximum reward that can be expected in the next state s_{t+1] after taking the action a in s_{t}

  v_next_states.volatile = False # Dont mess up the loss with volatile flag
  q_expected = rewards_given_for_action + (configuration.GAMMA * v_next_states)  # Compute the expected Q values, max_future_state_values may be 0 resulting in only giving the reward as the expected value of taking the V(s_{t+1}) action of state s_{t+1} which may be negative

  return F.smooth_l1_loss(q_of_actions_taken, q_expected)   # Compute Huber loss, Q value difference between what the model predicts as the value(how good) for taking action A in state S and what the expected Q value is for what taking that action(The actual reward given by plus environment the Q value of the next state)
