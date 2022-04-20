
import torch


class AdaptiveModel(torch.nn.Module):
    def __init__(self, in_features, num_categories):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=512, bias=True)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=self.fc1.out_features, out_features=num_categories, bias=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x
