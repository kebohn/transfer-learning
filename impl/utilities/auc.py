import torch
import numpy
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize


def softmax(x):
  return numpy.exp(x) / sum(numpy.exp(x))

def calculate_auc(features):
    y_test = []
    for ctn, (_, val) in enumerate(features.items()):
        key_list = numpy.repeat(ctn, val.size(0)) # add the same amount of category label to list as samples
        y_test.extend(key_list) # add the current category label list to whole label list

    # one hot encoding of test labels
    y = label_binarize(y_test, classes=list(set(y_test)))

    # combine features into one tensor
    features_tensor = torch.cat(tuple(features.values()), dim=0)

    # compute cosine similarity matrix
    sim = cosine_similarity(features_tensor.cpu())

    # ignore the similarity values on the diagonal because they will always be 1
    numpy.fill_diagonal(sim, 0.0)
    
    # create probabilty score matrix for each sample
    sim_proba = numpy.zeros((y.shape[0], y.shape[1]))

    current_pos = 0

    # iterate over each class
    for idx, (cat, val) in enumerate(features.items()):
    
        # get only similarity scores for one class
        sim_cat = sim[:, current_pos:(current_pos + val.size(0))]

        # get maximum value per feature sample (row)
        max_sim_cat = numpy.max(sim_cat, axis=1)

        # add the computed maximum similarity scores to already initialized probalibity matrix per sample and per current class
        sim_proba[:, idx] = max_sim_cat

        # increase current pos by number of features of the current class
        current_pos += val.size(0)

    # apply softmax on each row such that we have a probability for each class that sums up to 1
    sim_proba_softmax = numpy.apply_along_axis(softmax, 1, sim_proba)


    return roc_auc_score(y, sim_proba_softmax, multi_class='ovr', average='weighted')
