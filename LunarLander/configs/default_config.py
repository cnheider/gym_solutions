# coding=utf-8
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

import os

# General
CONFIG_NAME = __name__
RANDOM_SEED = 6
START_VISDOM_SERVER = True
VISDOM_SERVER = 'http://localhost'
if not START_VISDOM_SERVER:
  VISDOM_SERVER = 'http://visdom.ml'
CONNECT_TO_RUNNING_ENVIRONMENT = False
SPACER_SIZE = 60
SECONDS_IN_A_MINUTE = 60
USE_CUDA_IF_AVAILABLE = True
LOAD_PREVIOUS_MODEL_IF_AVAILABLE = True
RENDER_ENVIRONMENT = False
WINDOW_SIZE = 10

# Paths
DATA_SET = 'neodroid'
DATA_SET_DIRECTORY = os.path.join('/home/heider/Datasets', DATA_SET)
TARGET_FILE_NAME = 'target_position_rotation.csv'
DEPTH_IMAGES_DIRECTORY = 'depth'
MODEL_DIRECTORY = 'models'

# Training parameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.999
NUM_EPISODES = 1000
BATCH_SIZE = 60
SYNC_TARGET_MODEL_FREQUENCY = 1000
REPLAY_MEMORY_SIZE = 60000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.06
EPS_DECAY = 1000

# Architecture
ARCHITECTURE_CONFIGURATION = {
  'input_size'        : 0,
  'number_of_layers'   : 2,
  'hidden_layers_size': 50,
  'output_size'       : 0
}
