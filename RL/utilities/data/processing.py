# coding=utf-8
import numpy as np
import torch


def data_transform(observations, configuration, use_cuda):
  """

  :param use_cuda:
  :param observations:
  :param configuration:
  :return:
  """
  FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

  features = list(configuration.ARCHITECTURE_CONFIGURATION['input_size'])
  features = observations
  return FloatTensor([features])

def position_difference(pos1, pos2):
  """

  :param pos1:
  :param pos2:
  :return:
  """
  return (np.array(pos1) - np.array(pos2)).flatten()

def normalise_position(elements, bounds):
  """

  :param elements:
  :param bounds:
  :return:
  """
  normalised_0_1 = (np.array(elements) + np.array(bounds)) / (
    np.array(bounds) * 2)
  return normalised_0_1.flatten()
