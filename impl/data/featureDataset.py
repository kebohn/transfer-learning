import torch
import numpy

class FeatureDataset(torch.utils.data.Dataset):
  def __init__(self, features):
    self.features = features
    self.values = torch.cat(tuple(features.values()), dim=0) # combine features into one tensor
    self.names = []
    for key, val in features.items():
      key_list = numpy.repeat(key, val.size(1)) # add the same amount of category names to list as samples
      self.names.extend(key_list) # add the current category names list to whole names list


  def __len__(self):
    return len(self.names)


  def __getitem__(self, idx):
    name = self.names[idx] # retrieve name with idx because we have repeated the array we get the correct category name
    feature = self.values[idx, :] # retrieve feature tensor with idx because the tensor has been concacenated with all features
    names_set = list(set(self.names)) # construct a unique list with all category names
    label = names_set.index(name) # retrieve the index of the correct class this is automatically the label because of the preserving order
    return feature, label, name


  def get_categories(self):
    return len(list(self.features.keys()))a