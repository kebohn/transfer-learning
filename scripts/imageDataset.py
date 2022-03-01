import os, PIL
import pandas as pd
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(F"{img_dir}{annotations_file}")
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform


  def __len__(self):
    return len(self.img_labels)


  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = PIL.Image.open(img_path, 'r').convert('RGB') # open image skip transparency channel
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label

  
  def get_categories(self):
    return self.img_labels.iloc[-1, 1] + 1 # return number of categories