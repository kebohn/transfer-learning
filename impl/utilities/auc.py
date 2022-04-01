import torch
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt


def calculate_auc(features, test_features, equal=False):

  # combine features into one tensor
  features_tensor = torch.cat(tuple(features.values()), dim=0)
  features_test_tensor = torch.cat(tuple(test_features.values()), dim=0)

  # compute cosine similarity matrix
  sim = metrics.pairwise.cosine_similarity(features_test_tensor.detach().cpu(), features_tensor.detach().cpu())

  # ignore the similarity values on the diagonal because they will always be 1 in case we compare the features with themselves
  if equal:
    numpy.fill_diagonal(sim, 0.0)

  # counter variables
  current_pos = 0
  current_pos_test = 0

  # overall weighted auc score
  weighted_auc = 0.0

  # init calculated maximum similarity scores for positive and negative class for each sample
  y = numpy.zeros(sim.shape[0], dtype=int)

  # init calculated maximum similarity scores for positive and negative class 
  max_vals = numpy.zeros((sim.shape[0], 2))

  # iterate over each class
  for cat, val in test_features.items():

    end_pos_test = current_pos_test + val.size(0)
    end_pos = current_pos + features[cat].size(0)

    # define positive classes in samples
    y[:] = 0
    y[current_pos_test:end_pos_test] = 1

    # get only similarity scores for positive class
    sim_pos = sim[:,current_pos:end_pos]

    # get only similarity scores for negative class
    sim_neg = numpy.concatenate((sim[:,:current_pos], sim[:,end_pos:]), axis=1)

    # get maximum value positive class
    max_vals[:,0] = numpy.max(sim_pos, axis=1)

    # get maximum value negative class
    max_vals[:,1] = numpy.max(sim_neg, axis=1)

    # add auc score for current class to overall weighted score
    weighted_auc += metrics.roc_auc_score(y, max_vals[:,0]) * val.size(0)

    # increase current pos by number of features of the current class
    current_pos_test = end_pos_test
    current_pos = end_pos

  # weighted auc in case of inbalanced classes
  return weighted_auc / sim.shape[0]
