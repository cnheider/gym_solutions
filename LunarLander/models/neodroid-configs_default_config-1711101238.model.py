# coding=utf-8
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

import os

# General
CONFIG_NAME = __name__
CONFIG_FILE = __file__
RANDOM_SEED = 6
START_VISDOM_SERVER = True
VISDOM_SERVER = 'http://localhost'
if not START_VISDOM_SERVER:
  VISDOM_SERVER = 'http://visdom.ml'
CONNECT_TO_RUNNING_ENVIRONMENT = False
SPACER_SIZE = 60
SECONDS_IN_A_MINUTE = 60
USE_CUDA_IF_AVAILABLE = True
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = False
RENDER_ENVIRONMENT = False
WINDOW_SIZE = 20
#GYM_ENVIRONMENT = 'LunarLander-v2'
GYM_ENVIRONMENT = 'CartPole-v0'
#GYM_ENVIRONMENT = 'Pong-v0'
#GYM_ENVIRONMENT = 'Pong-ram-v0'

# Paths
DATA_SET = 'neodroid'
DATA_SET_DIRECTORY = os.path.join('/home/heider/Datasets', DATA_SET)
TARGET_FILE_NAME = 'target_position_rotation.csv'
DEPTH_IMAGES_DIRECTORY = 'depth'
MODEL_DIRECTORY = 'models'
CONFIG_DIRECTORY = 'configs'

# Training parameters
DOUBLE_DQN = False
LEARNING_RATE = 0.00025
WEIGHT_DECAY = 1e-5
NUM_EPISODES = 3000
BATCH_SIZE = 32
SYNC_TARGET_MODEL_FREQUENCY = 1000
REPLAY_MEMORY_SIZE = 100000
OBSERVATION_THRESHOLD = 5000
LEARNING_FREQUENCY = 4
DISCOUNT_FACTOR = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
CLIP_REWARD = False
CLAMP_GRADIENT = True
ALPHA = 0.95
EPS = 0.01

# Architecture
ARCHITECTURE_CONFIGURATION = {
  'input_size'        : 0,
  'hidden_layers': [256,128,64],
  'output_size'       : 0
}
