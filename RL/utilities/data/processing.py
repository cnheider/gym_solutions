# coding=utf-8
import numpy as np
import torch
from skimage import color, transform


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

def gray_downscale(state, use_cuda):
  LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
  FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
  StateTensorType = FloatTensor
  gray_img = color.rgb2gray(state)
  downsized_img = transform.resize(gray_img, (84, 84), mode='constant')
  state = torch.from_numpy(downsized_img).type(StateTensorType)  # 2D image tensor
  return torch.stack([state], 0).unsqueeze(0)