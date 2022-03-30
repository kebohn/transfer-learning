import matplotlib.pyplot as plt
import torch
import numpy


def save_feature_magnitude_hist(features):

  # compute feature magintudes
  feature_magnitudes =  {k:torch.linalg.vector_norm(v, ord=2, dim=1) for k,v in features.items()}
  magnitude_all_tensor = torch.cat(tuple(feature_magnitudes.values()), dim=0)
  bins = numpy.linspace(min(magnitude_all_tensor.detach().cpu()), max(magnitude_all_tensor.detach().cpu()), 100)

  # iterate over each class
  for idx, cat in enumerate(features.keys()):

    # construct one vs rest magnitudes
    feature_magnitudes_self = feature_magnitudes[cat].detach().cpu().numpy()
    feature_magnitudes_other = {k:v.detach().cpu().numpy() for k, v in feature_magnitudes.items() if k != cat}

    # combine all other magintues in one array
    feature_magnitudes_other = numpy.array(list(feature_magnitudes_other.values())).flatten()

    plt.figure()
    plt.hist(feature_magnitudes_self, bins, histtype='step', fill=False, label='class')
    plt.hist(feature_magnitudes_other, bins, histtype='step', fill=False, label='other')
    plt.savefig(F"hist{idx}.png")
    plt.close()