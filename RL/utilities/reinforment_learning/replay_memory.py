# coding=utf-8
"""
Description: ReplayMemory for storing transition tuples
Author: Christian Heider Nielsen
"""
import random
from collections import namedtuple

import configs.default_config as configuration

random.seed(configuration.RANDOM_SEED)

Transition = namedtuple('Transition',
                                 ('state',
                                  'action',
                                  'reward',
                                  'successor_states',
                                  'non_terminal'))


class ReplayMemory(object):
  """For storing transitions explored in the environment."""

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, *args):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(
          None)  # expand memory as needed, useful for sampling no None value state in sample()
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    """Randomly sample transitions from memory."""
    return random.sample(self.memory, batch_size)

  def __len__(self):
    """Return the length of the memory list."""
    return len(self.memory)
