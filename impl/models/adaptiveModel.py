
import torch


class AdaptiveModel(torch.nn.Module):
    def __init__(self, fc1_in, fc1_out, feature_size, category_size):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(in_features=fc1_in, out_features=fc1_out, bias=True)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=self.fc1.out_features, out_features=feature_size, bias=True)
        self.fc3 = torch.nn.Linear(in_features=self.fc2.out_features, out_features=category_size, bias=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(self.dropout(x))
        x = self.fc3(x)
        return x
