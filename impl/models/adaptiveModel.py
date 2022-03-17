
import torch

class AdaptiveModel(torch.nn.Module):
	def __init__(self, model, num_categories, shallow):
		super(AdaptiveModel, self).__init__()
		self.model = model # use input model as base model
		modules = list(self.model.children())[:-1] # remove last fully connected layer
		self.model = torch.nn.Sequential(*modules)

		# add shallow network on top of pre-trained base model
		if shallow: 
			self.classifier = torch.nn.Sequential(
				torch.nn.Linear(2048, 1024),
				torch.nn.Tanh(),
				torch.nn.Dropout(p=0.5),
				torch.nn.Linear(1024, 256),
				torch.nn.Linear(256, num_categories),
			) 
		
		# only add the classification layer on top of the pre-trained base model
		else:
			self.classifier = torch.nn.Sequential(
				torch.nn.Linear(2048, num_categories),
			)
			

	def forward(self, x):
		batch_size,_,_,_ = x.shape # get batch from input image
		x = self.model(x)
		x = torch.nn.functional.normalize(x, p=2, dim=0) # apply euclidean norm to the input before addit it into the adapter network
		x = x.reshape(batch_size, -1) # reshaping the batch size
		x = self.classifier(x)
		return x

