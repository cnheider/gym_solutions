"""
Description: Config for training
Author: Christian Heider Nielsen
"""

import os

# General
CONFIG_NAME = __name__
RANDOM_SEED = 42
START_VISDOM_SERVER = True
VISDOM_SERVER = 'http://localhost'
if not START_VISDOM_SERVER:
  VISDOM_SERVER = 'http://visdom.ml'
CONNECT_TO_RUNNING_ENVIRONMENT = True
SPACER_SIZE = 40
SECONDS_IN_A_MINUTE = 60

# Paths
DATA_SET = 'neodroid'
DATA_SET_DIRECTORY = os.path.join('/home/heider/Datasets', DATA_SET)
TARGET_FILE_NAME = 'target_position_rotation.csv'
DEPTH_IMAGES_DIRECTORY = 'depth'
MODEL_DIRECTORY = 'models'

# Training parameters
NUM_EPISODES = 200
BATCH_SIZE = 100
SYNC_TARGET_MODEL_FREQUENCY = 1000
MEMORY_SIZE = 200000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000


# Architecture
USE_ALL_DOFS = False
DOFS = 12
if not USE_ALL_DOFS:
  DOFS = 6
ARCHITECTURE_CONFIGURATION = {
  'input_size' : 4, #3 * 2 * 2 + 3,  # target(position,rotation,direction)
  # #gripper(position,rotation,direction)
  'number_of_layes': 3,
  'hidden_layers_size': 60,
  'output_size': DOFS+1  # gripper(+positional,+rotational,-positional,
  # -rotational)+nothing motions
}
