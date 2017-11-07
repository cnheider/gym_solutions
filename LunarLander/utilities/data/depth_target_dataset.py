import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DepthTargetPositionDataset(Dataset):
  """Face Landmarks dataset."""

  def __init__(self, csv_file, depth_image_dir, transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.target_coordinates_all = pd.read_csv(csv_file, header=None)
    self.depth_image_dir = depth_image_dir
    self.transform = transform

  def __len__(self):
    return len(self.target_coordinates_all)

  def __getitem__(self, idx):
    img_name = os.path.join(self.depth_image_dir, str(idx) + '.png')
    # image = io.imread(img_name)
    image = Image.open(img_name)
    image = image.convert("L")
    target_coordinates = self.target_coordinates_all.ix[idx,
                         :2].as_matrix().astype(float)
    # print(target_coordinates)
    # target_coordinates = target_coordinates.reshape(-1, 2)
    sample = {'image': image, 'target_coordinates': target_coordinates}

    if self.transform:
      image = self.transform(image)

    # return sample
    return image, target_coordinates
