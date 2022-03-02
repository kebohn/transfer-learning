import torch
from torchvision import models

class AdaptiveModel(torch.nn.Module):
	def __init__(self, num_categories):
		super(AdaptiveModel, self).__init__()
		self.model = models.resnet50(pretrained=True) # load pretrained model resnet-50
		self.model.fc = None # old classifier for pretrained imagenet, we don't use this here
		
		self.classifier = torch.nn.Sequential(
				torch.nn.Linear(2048, 512),
				torch.nn.BatchNorm1d(512),
				torch.nn.Dropout(0.2),
				torch.nn.Linear(512, 256),
				torch.nn.Linear(256, num_categories)
		)
			
	def forward(self, x):
		batch_size ,_,_,_ = x.shape # get batch from input image
		x = self.features(x) # see https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
		x = torch.nn.functional.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1) # reshaping the batch size
		x = self.classifier(x)
		return x

	def features(self, input):
		x = self.model.conv1(input)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)

		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)
		return x