import datetime
import time
from itertools import count

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

  widget_index_number_training = 0
  widget_index_number_evaluation = 0

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

    for episode_frame_number in count():
      if configuration.RENDER_ENVIRONMENT:
        environment.render()

      action_index = sample_action(state, model, steps_taken)
      observations, reward, terminated, _ = environment.step(action_index[0,0])#.cpu().numpy()[0, 0])
      steps_taken += 1

      state = FloatTensor([observations])

      next_state = None
      if not terminated:
        next_state = state

      reward_tensor = FloatTensor([reward]) # Convert to tensor

      memory.push(state, action_index, next_state, reward_tensor)

      state = next_state # Transition the before state to be 'next_state' in which the last action was execute and the observed

      if len(memory) >= configuration.BATCH_SIZE:

        transitions = memory.sample(configuration.BATCH_SIZE)  # Not necessarily consecutive

        loss = calculate_loss(model, target_model, transitions)
        print('Loss: {:.5f}'.format(loss.data[0]))

        optimiser.zero_grad()
        loss.backward()
        for param in model.parameters():
          param.grad.data.clamp_(-1, 1) # Clamp the gradient
        optimiser.step()

        if visualiser:
          (widget_index_number_evaluation,
          widget_index_number_training) = update_visualiser(
            visualiser, 'training',
            loss.data[0],
            loss.data[0],
            i_episode,
            widget_index_number_evaluation,
            widget_index_number_training)

      if steps_taken % configuration.SYNC_TARGET_MODEL_FREQUENCY == 0: # Update target model with the parameters of the learning model
        target_model.load_state_dict(model.state_dict())
        print('Target model synchronised')

      if terminated:
        episode_durations.append(episode_frame_number + 1)
        print('Interrupted')
        break

    print('-' * configuration.SPACER_SIZE)

  time_elapsed = time.time() - training_start_timestamp
  print('Training done, time elapsed: {:.0f}m {:.0f}s'.format(
      time_elapsed // configuration.SECONDS_IN_A_MINUTE,
      time_elapsed % configuration.SECONDS_IN_A_MINUTE))

  return model

def calculate_loss(model, target_model, transitions):
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

def main():
  _visualiser = Visdom(configuration.VISDOM_SERVER)

  _environment = gym.make('LunarLander-v2')

  configuration.ARCHITECTURE_CONFIGURATION['output_size'] = _environment.action_space.n
  configuration.ARCHITECTURE_CONFIGURATION['input_size']  = _environment.observation_space.shape[0]

  _model = MyDQN(configuration.ARCHITECTURE_CONFIGURATION)
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
