import neodroid as neo
import numpy as np
from neodroid.models import Reaction, Motion

env = neo.NeodroidEnvironment(name='gotolocation_transform.x86',
                              connect_to_running=False
                              )


def sample_motions(observations, step_size=0.1):
  try:
    target_pos = observations[b'GoalLocation'].get_position()
    gripper_pos = observations[b'TransformGripper'].get_position()

    dif_vec = np.array(gripper_pos) - np.array(target_pos)

    motions = [
      Motion('TransformGripper',
             'TransformGripperSingleAxisX',
             dif_vec[0] * step_size),
      Motion('TransformGripper',
             'TransformGripperSingleAxisY',
             dif_vec[1] * step_size),
      Motion('TransformGripper',
             'TransformGripperSingleAxisZ',
             dif_vec[2] * step_size),
      # Motion('RigidbodyGripper',
      #       'RigidbodyGripperRigidbodyRotX',
      #       sign * magnitude),
      # Motion('RigidbodyGripper',
      #       'RigidbodyGripperRigidbodyRotY',
      #       sign * magnitude),
      # Motion('RigidbodyGripper',
      #       'RigidbodyGripperRigidbodyRotZ',
      #       sign * magnitude)
    ]

    return motions
  except:
    return []


def sample_motions2(observations, step_size=0.01):
  try:
    target_pos = observations[b'GoalLocation'].get_position()
    gripper_pos = observations[b'TransformGripper'].get_position()

    dif_vec = np.array(gripper_pos) - np.array(target_pos)

    motions = [dif_vec[0] * step_size,
               dif_vec[1] * step_size,
               dif_vec[2] * step_size,
               ]
    return motions
  except:
    return []


_observations = env.step()  # Reaction(False, []))
_reaction = None
while 1:
  if _observations:
    _reaction = Reaction(False, sample_motions(_observations))
  else:
    _reaction = Reaction(False, [])
  _observations, _reward, _interrupted = env.step(_reaction)
  #print(_reward)
  print(_observations)
  if _interrupted:
    print('Interrupted')
