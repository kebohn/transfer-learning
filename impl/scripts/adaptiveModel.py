
import torch

class AdaptiveModel(torch.nn.Module):
	def __init__(self, model, num_categories):
		super(AdaptiveModel, self).__init__()
		self.model = model # use input model as base model
		modules = list(self.model.children())[:-1] # remove last fully connected layer
		self.model = torch.nn.Sequential(*modules)
		
		self.classifier = torch.nn.Sequential(
			#torch.nn.BatchNorm1d(2048), # Normalize output from pre-trained model
			torch.nn.Linear(2048, 512),
			torch.nn.Tanh(),
			torch.nn.BatchNorm1d(512),
			torch.nn.Linear(512, 256),
			torch.nn.Tanh(),
			torch.nn.BatchNorm1d(256),
			torch.nn.Linear(256, num_categories),
		)
			
	def forward(self, x):
		batch_size,_,_,_ = x.shape # get batch from input image
		x = self.model(x)
		x = x.reshape(batch_size, -1) # reshaping the batch size
		x = self.classifier(x)
		return x