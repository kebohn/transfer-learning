
import torch

class AdaptiveModel(torch.nn.Module):
	def __init__(self, num_categories):
		super(AdaptiveModel, self).__init__()

		self.fc1 = torch.nn.Linear(in_features=2048, out_features=120, bias=True)
		self.activation = torch.nn.Tanh()
		self.dropout = torch.nn.Dropout(p=0.5)
		self.fc2 = torch.nn.Linear(in_features=120, out_features=80, bias=True)
		self.fc3 = torch.nn.Linear(in_features=80, out_features=num_categories, bias=True)

		
	def forward(self, x):
		x = self.activation(self.fc1(x))
		x = self.fc2(self.dropout(x))
		x = self.fc3(x)
		return x

	def reset_weights(self):
		for layer in self.model.children():
			if hasattr(layer, 'reset_parameters'):
					layer.reset_parameters()
