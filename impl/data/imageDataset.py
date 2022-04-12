import os, PIL
import pandas as pd
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, samples=0):
    self.img_labels = pd.read_csv(F"{img_dir}{annotations_file}")
    if samples > 0:
      # extract specified amount of samples per category
      self.img_labels = self.img_labels.groupby('category').head(samples).reset_index(drop=True)
    self.img_dir = img_dir
    self.transform = transform


  def __len__(self):
    return len(self.img_labels)


  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = PIL.Image.open(img_path, 'r').convert('RGB') # open image skip transparency channel
    label = self.img_labels.iloc[idx, 1]
    name = self.img_labels.iloc[idx, 2]
    if self.transform:
        image = self.transform(image)
    return image, label, name, self.img_labels.iloc[idx, 0]

  
  def get_categories(self):
    return self.img_labels['category'].nunique() # count unique values in category column