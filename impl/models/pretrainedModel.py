
import torch

class PretrainedModel(torch.nn.Module):
	def __init__(self, model, num_categories):
		super(PretrainedModel, self).__init__()
		
		# use input model as base model
		self.model = model

		# remove last fully connected layer
		modules = list(self.model.children())[:-1]
		self.model = torch.nn.Sequential(*modules)

		# only add the classification layer on top of the pre-trained base model
		self.classifier = torch.nn.Sequential(torch.nn.Linear(2048, num_categories))

	def forward(self, x):
		batch_size,_,_,_ = x.shape # get batch from input image
		x = self.model(x)
		x = x.reshape(batch_size, -1) # reshaping the batch size
		x = self.classifier(x)
		return x
