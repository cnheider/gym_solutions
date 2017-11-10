# coding=utf-8
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

import os

# CONSTANTS
WINDOW_SIZE = 20
SPACER_SIZE = 60
SECONDS_IN_A_MINUTE = 60
RANDOM_SEED = 6

# General
CONFIG_NAME = __name__
CONFIG_FILE = __file__
USE_CUDA_IF_AVAILABLE = True

# Visualisation
START_VISDOM_SERVER = True
VISDOM_SERVER = 'http://localhost'
if not START_VISDOM_SERVER:
  VISDOM_SERVER = 'http://visdom.ml'

# Paths
DATA_SET = 'neodroid'
DATA_SET_DIRECTORY = os.path.join('/home/heider/Datasets', DATA_SET)
TARGET_FILE_NAME = 'target_position_rotation.csv'
DEPTH_IMAGES_DIRECTORY = 'depth'
MODEL_DIRECTORY = 'models'
CONFIG_DIRECTORY = 'configs'

# Environment parameters
CONNECT_TO_RUNNING_ENVIRONMENT = False
RENDER_ENVIRONMENT = False
#GYM_ENVIRONMENT = 'LunarLander-v2'
GYM_ENVIRONMENT = 'CartPole-v0'
#GYM_ENVIRONMENT = 'Pong-v0'
#GYM_ENVIRONMENT = 'Pong-ram-v0'
NUM_EPISODES = 10000

# Epsilon random action parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000

# Training parameters
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False
DOUBLE_DQN = True
CLIP_REWARD = True
CLAMP_GRADIENT = True
BATCH_SIZE = 32
LEARNING_FREQUENCY = 4
SYNC_TARGET_MODEL_FREQUENCY = 10000
REPLAY_MEMORY_SIZE = 1000000
INITIAL_OBSERVATION_PERIOD = 1000
DISCOUNT_FACTOR = 0.99

# Optimiser
LEARNING_RATE = 0.00025
WEIGHT_DECAY = 1e-5
ALPHA = 0.95
EPSILON = 0.01

# Architecture
ARCHITECTURE_CONFIGURATION = {
  'input_size'        : 0,
  'hidden_layers': [16,32,16],
  'output_size'       : 0
}
