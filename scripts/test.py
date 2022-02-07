import torch
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

data = torch.randn(100, 10)
test = torch.randn(1, 10)

dist = torch.norm(data - test, dim=1, p=None)
knn = dist.topk(3, largest=False)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))


