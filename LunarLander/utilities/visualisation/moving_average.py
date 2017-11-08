# coding=utf-8
import numpy as np

class StatisticAggregator:
  """

  """

  def __init__(self, window_size=100):
    self.values = [0 for i in range(window_size+9)]

  def append(self, val):
    self.values.append(val)

  def moving_average(self, window_size=100):
    return sum(self.values[-window_size:])/window_size

  def get_size(self):
    return self.count