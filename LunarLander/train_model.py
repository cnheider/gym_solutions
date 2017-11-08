import glob
import datetime
import time
from itertools import count
import os
import math
import random

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
from utilities.reinforment_learning.loss import calculate_loss

from architectures.dqn import MyDQN
import configs.default_config as configuration

ImageFile.LOAD_TRUNCATED_IMAGES = True

_use_cuda = False
if configuration.USE_CUDA_IF_AVAILABLE:
  _use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor

def training_loop(model,
                  target_model,
                  environment,
                  visualiser=None):
  windows={}
  steps_taken = 0

  episode_durations = []
  memory = ReplayMemory(configuration.REPLAY_MEMORY_SIZE)

  optimiser = optimisation.Adam(model.parameters(), lr=configuration.LEARNING_RATE)

  training_start_timestamp = time.time()
  print('-' * configuration.SPACER_SIZE)
  for i_episode in range(configuration.NUM_EPISODES):
    print('Episode {}/{}'.format(i_episode, configuration.NUM_EPISODES - 1))

    observations = environment.reset()   # Initial state
    state = FloatTensor([observations])
    reward = 0
    action = 0
    terminated = False

    for episode_frame_number in count():
      if configuration.RENDER_ENVIRONMENT:
        environment.render()

      sample = random.random()
      eps_threshold = configuration.EPS_END + ((configuration.EPS_START - configuration.EPS_END) * math.exp(-1. * steps_taken / configuration.EPS_DECAY))
      if sample > eps_threshold:
        action = sample_action(state, model, steps_taken)
        observations, reward, terminated, _ = environment.step(action[0,0])
      else:
        action = environment.action_space.sample()
        observations, reward, terminated, _ = environment.step(action)
        action = LongTensor([[action]])
      steps_taken += 1

      state = FloatTensor([observations])

      next_state = None
      if not terminated:
        next_state = state

      reward_tensor = FloatTensor([reward]) # Convert to tensor

      memory.push(state, action, next_state, reward_tensor)

      state = next_state # Transition the before state to be 'next_state' in which the last action was execute and the observed

      loss=0
      if len(memory) >= configuration.BATCH_SIZE:

        transitions = memory.sample(configuration.BATCH_SIZE)  # Not necessarily consecutive

        loss = calculate_loss(model, target_model, transitions, configuration)

        optimiser.zero_grad()
        loss.backward()
        for param in model.parameters():
          param.grad.data.clamp_(-1, 1) # Clamp the gradient
        optimiser.step()

      if steps_taken % configuration.SYNC_TARGET_MODEL_FREQUENCY == 0: # Update target model with the parameters of the learning model
        target_model.load_state_dict(model.state_dict())
        print('*** Target model synchronised ***')

      if terminated:
        print('Terminated')
        episode_length = episode_frame_number + 1
        episode_durations.append(episode_length)
        rgb_array = environment.render(mode='rgb_array').swapaxes(0,2).swapaxes(1,2)
        if visualiser:
          windows = update_visualiser(
          visualiser,
          i_episode,
          loss.data[0],
          episode_length,
          rgb_array,
          windows)
        break

    print('-' * configuration.SPACER_SIZE)

  time_elapsed = time.time() - training_start_timestamp
  print('Training done, time elapsed: {:.0f}m {:.0f}s'.format(
      time_elapsed // configuration.SECONDS_IN_A_MINUTE,
      time_elapsed % configuration.SECONDS_IN_A_MINUTE))

  return model

def main():
  _visualiser = Visdom(configuration.VISDOM_SERVER)

  _environment = gym.make('LunarLander-v2')
  _environment.seed(configuration.RANDOM_SEED)

  configuration.ARCHITECTURE_CONFIGURATION['input_size']  = _environment.observation_space.shape[0] # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
  print('observation dimensions: ', _environment.observation_space.shape[0])

  configuration.ARCHITECTURE_CONFIGURATION['output_size'] = _environment.action_space.n
  print('action dimensions: ', _environment.action_space.n)

  _model = MyDQN(configuration.ARCHITECTURE_CONFIGURATION)
  if configuration.LOAD_PREVIOUS_MODEL_IF_AVAILABLE:
    _list_of_files = glob.glob(configuration.MODEL_DIRECTORY + '/*')
    _latest_model = max(_list_of_files, key=os.path.getctime)
    print('loading previous model: ' + _latest_model)
    _model = torch.load(_latest_model)
  _target_model = MyDQN(configuration.ARCHITECTURE_CONFIGURATION)
  _target_model.load_state_dict(_model.state_dict())

  if _use_cuda:
    _model = _model.cuda()
    _target_model.cuda()

  _target_model = training_loop(_model,
                                _target_model,
                                _environment,
                                _visualiser)

  _model_date = datetime.datetime.now()
  _model_name = '{}-{}-{}.model'.format(configuration.DATA_SET,
                                        configuration.CONFIG_NAME.replace('.', '_'),
                                        _model_date.strftime('%y%m%d%H%M'))
  torch.save(_model, os.path.join(configuration.MODEL_DIRECTORY, _model_name))

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
        'Keeping visdom running, pressing enter will terminate visdom process..')
    VISDOM_PROCESS.terminate()
