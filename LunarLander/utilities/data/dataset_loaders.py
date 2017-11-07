import os

import torch
from perfect_information_navigator.utilities.depth_target_dataset import \
  DepthTargetPositionDataset
from perfect_information_navigator.utilities.processing import data_transforms


def get_dataset_loaders(data_set_directory, target_file_name,
                        depth_images_directory, batch_size,
                        num_data_loader_workers):
  datasets = {
    x: DepthTargetPositionDataset(
        os.path.join(
            os.path.join(data_set_directory, x), target_file_name),
        os.path.join(
            os.path.join(data_set_directory, x), depth_images_directory),
        data_transforms[x])
    for x in ['training', 'evaluation']
  }

  dataset_loaders = {
    x: torch.utils.data.DataLoader(
        datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_workers)
    for x in ['training', 'evaluation']
  }

  dataset_sizes = {x: len(datasets[x]) for x in ['training', 'evaluation']}

  return dataset_loaders, dataset_sizes
