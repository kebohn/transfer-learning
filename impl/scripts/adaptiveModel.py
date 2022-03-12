
import torch

class AdaptiveModel(torch.nn.Module):
	def __init__(self, model, num_categories, shallow, vis):
		super(AdaptiveModel, self).__init__()
		self.model = model # use input model as base model
		modules = list(self.model.children())[:-1] # remove last fully connected layer
		self.model = torch.nn.Sequential(*modules)

		if vis is not None:
			self.__extractConvLayers(modules)

		# add shallow network on top of pre-trained base model
		if shallow: 
			self.classifier = torch.nn.Sequential(
				torch.nn.BatchNorm1d(2048), # Normalize output from pre-trained base model
				torch.nn.Linear(2048, 1024),
				torch.nn.BatchNorm1d(1024),
				torch.nn.Tanh(),
				torch.nn.Dropout(p=0.5),
				torch.nn.Linear(1024, 256),
				torch.nn.BatchNorm1d(256),
				torch.nn.Tanh(),
				torch.nn.Dropout(p=0.5),
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
		x = x.reshape(batch_size, -1) # reshaping the batch size
		x = self.classifier(x)
		return x


	def __extractConvLayers(self, modules):
		self.layers = []
		for m in modules:
			if type(m) == torch.nn.Conv2d:
				self.layers.append(m)
			elif type(m) == torch.nn.Sequential:
				for i in m.children:
					if type(m) == torch.nn.Conv2d:
						self.layers.append(m)


	def get_feature_maps(self, input):
		res = [self.layers[0](input)] # first output
		# consequently use the previous output as input for the next layer
		for l in range(1, len(self.layers)):
			res.append(self.layers[l](res[-1]))
		return res

	def get_conv_layers(self):
		return self.layers
