import torch

class IdentityModel(torch.nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        
    def forward(self, x):
        return x
