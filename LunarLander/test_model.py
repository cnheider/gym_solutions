import glob
import datetime
import time
from itertools import count
import os

import torch
import torch.optim as optimisation
from PIL import ImageFile
from torch.autograd import Variable
from visdom import Visdom
import neodroid as neo
import torch.nn.functional as F
import gym

from utilities.data.processing import data_transform
from utilities.visualisation import update_visualiser

from utilities.reinforment_learning.action import testing_sample_action
from utilities.reinforment_learning.filtering import is_non_terminal
from utilities.reinforment_learning.replay_memory import (
  ReplayMemory,
  TransitionQuadruple)

from architectures.dqn import MyDQN
import configs.default_config as configuration

ImageFile.LOAD_TRUNCATED_IMAGES = True

_use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor

def testing_loop(model,
                  environment,
                  visualiser=None):

  widget_index_number_training = None
  widget_index_number_evaluation = None

  episode_durations = []

  print('-' * configuration.SPACER_SIZE)
  for i_episode in range(configuration.NUM_EPISODES):
    print('Episode {}/{}'.format(i_episode, configuration.NUM_EPISODES - 1))

    observations = environment.reset()   # Initial state

    state = FloatTensor([observations])

    for episode_frame_number in count():
      if configuration.RENDER_ENVIRONMENT:
        environment.render()

      action_index = testing_sample_action(state, model)
      observations, reward, terminated, _ = environment.step(action_index[0,0])
      state = FloatTensor([observations])

      if visualiser:
        (widget_index_number_evaluation,
        widget_index_number_training) = update_visualiser(
          visualiser, 'training',
          reward,
          reward,
          i_episode,
          widget_index_number_evaluation,
          widget_index_number_training)

      if terminated:
        episode_durations.append(episode_frame_number + 1)
        print('Interrupted')
        break

    print('-' * configuration.SPACER_SIZE)

def main():
  _visualiser = Visdom(configuration.VISDOM_SERVER)

  _environment = gym.make('LunarLander-v2')

  _list_of_files = glob.glob(configuration.MODEL_DIRECTORY + '/*')
  _latest_model = max(_list_of_files, key=os.path.getctime)

  print('loading lastest model: ' + _latest_model)

  _model = torch.load(_latest_model)

  if _use_cuda:
    _model = _model.cuda()

  testing_loop(_model, _environment, _visualiser)

if __name__ == '__main__':
  main()
