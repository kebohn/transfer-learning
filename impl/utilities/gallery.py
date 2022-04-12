import torch
from sklearn import metrics
import numpy


def init_k_best_gallery(features, k): 
  '''Returns the k most similar feature samples per category by computing the similarity distance'''

  # create feature tensor
  features_tensor = torch.cat(tuple(features.values()), dim=0)

  # compute cosine similarity matrix
  sim = metrics.pairwise.cosine_similarity(features_tensor.detach().cpu())

  # ignore similarity values from same sample
  numpy.fill_diagonal(sim, 0.0)

  gallery = {}
  training_features = {}

  # counter variable
  current_pos = 0
 
  for cat, value in features.items():

    sample_size = value.size(0)

    # get only similarity score from same category
    sim_cat = sim[current_pos:current_pos + sample_size,current_pos:current_pos + sample_size]

    if k > 1:
      # get the k-1 indices with largest scores along each row aka feature
      k_max = numpy.argpartition(-sim_cat, k-1, axis=1)[:,k-2::-1]

      # define row indices with k-1 repeats 
      row_idx = numpy.arange(sample_size)
      row_idx = numpy.repeat(row_idx, k-1)

      # get all respective scores
      scores = sim_cat[row_idx, k_max.ravel()].reshape(sample_size, k-1)

      # compute mean score per feature sample with the best k-1 scores
      mean_scores = numpy.mean(scores, axis=1)
    
      # get idx with max mean score - this is automatically the best sample 
      max_idx = numpy.argmax(mean_scores)

      # retrieve the indices which resulted in the best k-1 scores
      best_k_indices = k_max[max_idx, :]

      # append the best row index to list
      best_k_indices = numpy.append(best_k_indices, max_idx)
    else:
      # compute mean score per feature sample
      mean_scores = numpy.mean(sim_cat, axis=1)

      # get idx with max mean score - this is automatically the best sample 
      best_k_indices = numpy.argmax(mean_scores)
    
    # add the chosen features to the gallery
    gallery[cat] = value[best_k_indices,:]

    # the remaining features are for training
    all_idx = numpy.arange(sample_size)
    remaining_idx = numpy.delete(all_idx, best_k_indices)
    training_features[cat] = value[remaining_idx,:]

    current_pos += sample_size

  return gallery, training_features