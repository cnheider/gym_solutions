# coding=utf-8
import datetime
import glob
import os
import time
from itertools import count
import shutil

import gym
import torch
import torch.optim as optimisation
from PIL import ImageFile
from torch.autograd import Variable
from visdom import Visdom
import numpy as np
import torch.nn.functional as F

import configs.default_config as configuration
from architectures.mlp import LinearOutputAffineMLP
from utilities.reinforment_learning.action import sample_action, \
  epsilon_random
from utilities.reinforment_learning.optimisation import optimise_model
from utilities.reinforment_learning.replay_memory import (
  ReplayMemory, TransitionQuadruple)
from utilities.visualisation import update_visualiser
from utilities.visualisation.moving_average import StatisticAggregator

ImageFile.LOAD_TRUNCATED_IMAGES = True

_use_cuda = False
if configuration.USE_CUDA_IF_AVAILABLE:
  _use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor
StateTensorType = FloatTensor
ActionTensorType = LongTensor

torch.manual_seed(configuration.RANDOM_SEED)


def training_loop(model,
                  target_model,
                  environment,
                  visualiser=None):
  """

  :param model:
  :param target_model:
  :param environment:
  :param visualiser:
  :return:
  """
  # Statistics
  windows = {}

  total_steps_taken = 0
  episode_losses = StatisticAggregator(configuration.WINDOW_SIZE)
  episode_rewards = StatisticAggregator(configuration.WINDOW_SIZE)
  episode_durations = StatisticAggregator(configuration.WINDOW_SIZE)
  memory = ReplayMemory(configuration.REPLAY_MEMORY_SIZE)

  #optimiser = optimisation.Adam(model.parameters(),
  #                              lr=configuration.LEARNING_RATE)
                                #weight_decay=configuration.WEIGHT_DECAY)

  optimiser = optimisation.RMSprop(model.parameters(),
                                   lr=configuration.LEARNING_RATE,
                                   eps=configuration.EPSILON,
                                   alpha=configuration.ALPHA)
  #                                 weight_decay=configuration.WEIGHT_DECAY)

  training_start_timestamp = time.time()
  print('-' * configuration.SPACER_SIZE)
  for episode_i in range(configuration.NUM_EPISODES):
    print('Episode {}/{} | Total steps taken {}'.format(episode_i,
                                         configuration.NUM_EPISODES - 1,
                                          total_steps_taken))

    episode_loss = 0
    episode_reward = 0

    observations = environment.reset()  # Initial state
    state = StateTensorType([observations])

    for episode_frame_number in count():
      if configuration.RENDER_ENVIRONMENT:
        environment.render()

      # Sample action based on the state from last iteration and take a step
      action = sample_action(environment,
                             model,
                             state,
                             configuration,
                             total_steps_taken)
      observations, reward, terminated, _ = environment.step(action[0, 0])

      if configuration.CLIP_REWARD:
        reward = max(-1.0, min(reward, 1.0)) # Reward clipping

      # Convert to tensors
      reward_tensor = FloatTensor([reward])
      non_terminal_tensor = ByteTensor([not terminated])

      # If environment terminated then there is no next state
      future = None
      if not terminated:
        future = StateTensorType([observations])

      # Transition the before state to be 'future'
      # in which the last action was execute and the observed
      memory.push(state, action, reward_tensor, future, non_terminal_tensor)

      state = future

      loss = 0
      if len(memory) >= configuration.BATCH_SIZE and \
              total_steps_taken > configuration.INITIAL_OBSERVATION_PERIOD and \
          total_steps_taken % configuration.LEARNING_FREQUENCY == 0:
        random_transitions = memory.sample(configuration.BATCH_SIZE)

        loss = optimise_model(model,
                                      target_model,
                                      optimiser,
                                      random_transitions,
                                      configuration,
                                      _use_cuda)

      # Update target model with the parameters of the learning model
      if total_steps_taken % configuration.SYNC_TARGET_MODEL_FREQUENCY == 0 \
          and configuration.DOUBLE_DQN:
        target_model.load_state_dict(model.state_dict())
        print('*** Target model synchronised ***')

      total_steps_taken += 1
      episode_reward += reward
      episode_loss += loss

      if terminated:
        episode_length = episode_frame_number + 1

        episode_losses.append(episode_loss)
        episode_rewards.append(episode_reward)
        episode_durations.append(episode_length)

        rgb_array = environment.render(mode='rgb_array')
        if visualiser:
          windows = update_visualiser(
              visualiser,
              episode_i,
              episode_loss / episode_length,
              episode_losses.moving_average(configuration.WINDOW_SIZE),
              episode_reward / episode_length,
              episode_rewards.moving_average(configuration.WINDOW_SIZE),
              episode_length,
              episode_durations.moving_average(configuration.WINDOW_SIZE),
              rgb_array.swapaxes(0, 2).swapaxes(1, 2),
              windows,
              configuration)
        print('Episode terminated')
        break

    print('-' * configuration.SPACER_SIZE)

  time_elapsed = time.time() - training_start_timestamp
  print('Training done, time elapsed: {:.0f}m {:.0f}s'.format(
      time_elapsed // configuration.SECONDS_IN_A_MINUTE,
      time_elapsed % configuration.SECONDS_IN_A_MINUTE))

  target_model.load_state_dict(model.state_dict())
  return target_model


def main():
  """

  :return:
  """
  _visualiser = Visdom(configuration.VISDOM_SERVER)

  _environment = gym.make(configuration.GYM_ENVIRONMENT)
  _environment.seed(configuration.RANDOM_SEED)

  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
  configuration.ARCHITECTURE_CONFIGURATION['input_size'] = \
    _environment.observation_space.shape[0]
  print('observation dimensions: ', _environment.observation_space.shape[0])

  configuration.ARCHITECTURE_CONFIGURATION[
    'output_size'] = _environment.action_space.n
  print('action dimensions: ', _environment.action_space.n)

  _model = LinearOutputAffineMLP(configuration.ARCHITECTURE_CONFIGURATION)
  if configuration.LOAD_PREVIOUS_MODEL_IF_AVAILABLE:
    _list_of_files = glob.glob(configuration.MODEL_DIRECTORY + '/*')
    _latest_model = max(_list_of_files, key=os.path.getctime)
    print('loading previous model: ' + _latest_model)
    _model = torch.load(_latest_model)
  _target_model = LinearOutputAffineMLP(configuration.ARCHITECTURE_CONFIGURATION)
  _target_model.load_state_dict(_model.state_dict())

  if _use_cuda:
    _model = _model.cuda()
    _target_model.cuda()

  _target_model = training_loop(_model,
                                _target_model,
                                _environment,
                                _visualiser)

  _environment.render(close=True)
  _environment.close()

  _model_date = datetime.datetime.now()
  _model_name = '{}-{}-{}.model'.format(configuration.DATA_SET,
                                        configuration.CONFIG_NAME.replace('.',
                                                                          '_'),
                                        _model_date.strftime('%y%m%d%H%M'))
  _model_path = os.path.join(configuration.MODEL_DIRECTORY, _model_name)
  torch.save(_target_model, _model_path)

  shutil.copyfile(os.path.join(configuration.CONFIG_DIRECTORY,
                               configuration.CONFIG_FILE),
                  _model_path+'.py')




if __name__ == '__main__':
  VISDOM_PROCESS = None
  if configuration.START_VISDOM_SERVER:
    print("Starting visdom server process")
    import subprocess

    VISDOM_PROCESS = subprocess.Popen(
        ['python3', 'utilities/visualisation/run_visdom_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

  main()

  if VISDOM_PROCESS:
    input(
        'Keeping visdom running, pressing '
        'enter will terminate visdom process..')
    VISDOM_PROCESS.terminate()
