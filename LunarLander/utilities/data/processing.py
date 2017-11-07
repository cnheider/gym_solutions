import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from sklearn import preprocessing

_use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor

def data_transform(observations, configuration):
  features = [configuration.ARCHITECTURE_CONFIGURATION['input_size']]
  if observations:
    gripper = observations[b'TransformGripper']
    goal = observations[b'GoalLocation']

    obstruction_names= [b'Wall',b'Wall (1)',
                        b'Wall (2)',b'Wall (3)',
                        b'Sphere',b'Cube']
    distance_to_nearest_obstacle=1000.0
    for name in obstruction_names:
      dista = np.linalg.norm(position_difference(gripper.get_position(),
                                            observations[
        name].get_position()))
      if dista < distance_to_nearest_obstacle:
        distance_to_nearest_obstacle = dista

    bounding_box = observations[b'Ground']
    bounds = [5.5, 3, 5.5]
    #features = [
      #normalise_position(
    #    position_difference(gripper.get_position(), goal.get_position())#,
    #    bounds #bounding_box.get_data()
    #    )
    features = preprocessing.normalize(position_difference(
        gripper.get_position(),
                                                goal.get_position(

                                                )).reshape(-1, 1),
                            norm='l2').flatten().tolist()
    #]
    features += [distance_to_nearest_obstacle]

    #features += [observation.get_rotation() for observation in observations.values()]
    #features += [observation.get_direction() for observation in observations.values()]

  features = np.array(features).flatten().tolist()
  print(features)
  return FloatTensor([features])

def position_difference(pos1, pos2):
  return (np.array(pos1) - np.array(pos2)).flatten()

def normalise_position(elements, bounds):
  normalised_0_1 = (np.array(elements)+np.array(bounds))/(np.array(bounds)*2)
  return (normalised_0_1).flatten()