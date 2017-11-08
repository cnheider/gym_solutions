# coding=utf-8
import glob
import os
from itertools import count

import gym
import torch

import configs.default_config as configuration
from utilities.reinforment_learning.action import sample_action

_use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor


def testing_loop(model,
                 environment):
  episode_durations = []

  print('-' * configuration.SPACER_SIZE)
  for i_episode in range(configuration.NUM_EPISODES):
    print('Episode {}/{}'.format(i_episode, configuration.NUM_EPISODES - 1))

    observations = environment.reset()  # Initial state

    state = FloatTensor([observations])

    for episode_frame_number in count():
      if configuration.RENDER_ENVIRONMENT:
        environment.render()

      action_index = sample_action(state, model)
      observations, reward, terminated, _ = environment.step(
          action_index[0, 0])
      state = FloatTensor([observations])

      if terminated:
        episode_durations.append(episode_frame_number + 1)
        print('Interrupted')
        break

    print('-' * configuration.SPACER_SIZE)


def main():
  """

  """

  _environment = gym.make('LunarLander-v2')

  _list_of_files = glob.glob(configuration.MODEL_DIRECTORY + '/*')
  _latest_model = max(_list_of_files, key=os.path.getctime)

  print('loading latest model: ' + _latest_model)

  _model = torch.load(_latest_model)

  if _use_cuda:
    _model = _model.cuda()

  testing_loop(_model, _environment)


if __name__ == '__main__':
  main()
