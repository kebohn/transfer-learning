import torch
import numpy


class FeatureDataset(torch.utils.data.Dataset):
	def __init__(self, features):
		# sort keys such that the class label is correct
		self.features = dict(sorted(features.items()))
		# combine features into one tensor
		self.values = torch.cat(tuple(features.values()), dim=0)
		self.names = []
		for key, val in features.items():
			# add the same amount of category names to list as samples
			key_list = numpy.repeat(key, val.size(0))
			# add the current category names list to whole names list
			self.names.extend(key_list)

	def __len__(self):
		return len(self.names)

	def __getitem__(self, idx):
		# retrieve name with idx because we have repeated the array we get the correct category name
		name = self.names[idx]
		# retrieve feature tensor with idx because the tensor has been concatenated with all features
		feature = self.values[idx, :]
		# construct a unique list with all category names
		names_set = list(numpy.unique(numpy.array(self.names)))
		# retrieve the index of the correct class this is automatically the label because of the preserving order
		label = names_set.index(name)
		return feature, label, name


  def get_categories(self):
    return len(list(self.features.keys()))