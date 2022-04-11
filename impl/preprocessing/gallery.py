import torch
from sklearn import metrics
import numpy


def best_k_samples(k, features): 

  # create feature tensor
  features_tensor = torch.cat(tuple(features.values()), dim=0)

  # compute cosine similarity matrix
  sim = metrics.pairwise.cosine_similarity(features_tensor.detach().cpu())

  # ignore similarity values from same sample
  numpy.fill_diagonal(sim, 0.0)

  # counter variable
  current_pos = 0
 
  for cat, value in features.items():

    sim_cat = sim[current_pos:current_pos + value.size(0),current_pos:current_pos + value.size(0)]

    if k > 1:
      # get the k-1 indices with largest scores along each row aka feature
      k_max = numpy.argpartition(-sim_cat, k-1, axis=1)[:,k-2::-1]

      # define row indices with k-1 repeats 
      row_idx = numpy.arange(value.size(0))
      row_idx = numpy.repeat(row_idx, k-1)

      # get all respective scores
      scores = sim_cat[row_idx, k_max.ravel()].reshape(value.size(0), k-1)

      # compute mean score per feature sample with the best k-1 scores
      mean_scores = numpy.mean(scores, axis=1)
    
      # get idx with max mean score - this is automatically the best sample 
      max_idx = numpy.argmax(mean_scores)

      # retrieve the indices which resulted in the best k-1 scores
      best_k_samples = k_max[max_idx, :]

      # append the best row index to list
      best_k_samples = numpy. append(best_k_samples, max_idx)

      print(best_k_samples)
    else:
      # compute mean score per feature sample
      mean_scores = numpy.mean(sim_cat, axis=1)

      # get idx with max mean score - this is automatically the best sample 
      best_k_samples = numpy.argmax(mean_scores)
    
    print(best_k_samples)

    current_pos += value.size(0)


features = torch.load('/local/scratch/bohn/results/indoor/pre_trained_extraction/cosine_new/features_train_size_5.pt')

best_k_samples(1, features)