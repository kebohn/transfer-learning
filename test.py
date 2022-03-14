import torchvision
import torch
import matplotlib.pyplot as plt
import numpy

features = torch.load("/local/scratch/bohn/impl/scripts/5_features.pt")


for key, value in features.items():
    print(value.size())