import torch
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt
import utilities


def score_iter(features, features_test, equal):
  # combine features into one tensor
  features_tensor = torch.cat(tuple(features.values()), dim=0)
  features_test_tensor = torch.cat(tuple(features_test.values()), dim=0)

  # compute cosine similarity matrix
  sim = metrics.pairwise.cosine_similarity(features_test_tensor.detach().cpu(), features_tensor.detach().cpu())

  # ignore the similarity values on the diagonal because they will always be 1 in case we compare the features with themselves
  if equal:
    numpy.fill_diagonal(sim, 0.0)

  # counter variables
  current_pos = 0
  current_pos_test = 0

  # init calculated maximum similarity scores for positive and negative class for each sample
  y = numpy.zeros(sim.shape[0], dtype=int)

  # init calculated maximum similarity scores for positive and negative class 
  max_vals = numpy.zeros((sim.shape[0], 2))

  # iterate over each class
  for cat, val in features_test.items():

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

    # increase current pos by number of features of the current class
    current_pos_test = end_pos_test
    current_pos = end_pos

    yield y, max_vals, val.size(0), sim.shape[0]


def calculate_auc(features, features_test, equal=False):

  # overall weighted auc score
  weighted_auc = 0.0

  for y, max_vals, class_size, size in score_iter(features, features_test, equal):

    # add auc score for current class to overall weighted score
    weighted_auc += metrics.roc_auc_score(y, max_vals[:,0]) * class_size

  # weighted auc in case of inbalanced classes
  return weighted_auc / size


def save_roc_curve(name, fpr, tpr, roc_auc):
  plt.figure()
  lw = 2
  plt.plot(
      fpr,
      tpr,
      color="darkorange",
      lw=lw,
      label=F"ROC curve (area = {roc_auc:.2f})",
  )
  plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Indoor")
  plt.legend(loc="lower right")
  plt.savefig(name)
  plt.close()


def perform_roc(features, features_test, equal=False):

  # plot roc curve
  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  for i, (y, max_vals, class_size, size) in enumerate(score_iter(features, features_test, equal)):

    # compute false positive rate and true positive rate for each class
    fpr[i], tpr[i], _ = metrics.roc_curve(y, max_vals[:,0])

    # compute area under the curve for each class
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    save_roc_curve(F'roc_class_{i}.png', fpr[i], tpr[i], roc_auc[i])

  # First aggregate all false positive rates
  all_fpr = numpy.unique(numpy.concatenate([fpr[j] for j in range(i)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = numpy.zeros_like(all_fpr)
  for j in range(i):
      mean_tpr += numpy.interp(all_fpr, fpr[j], tpr[j])

  # Finally average it and compute AUC
  mean_tpr /= i

  makro_auc = metrics.auc(all_fpr, mean_tpr)

  save_roc_curve(F'roc_makro.png', all_fpr, mean_tpr, makro_auc)

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = makro_auc

  # TODO: compute weighted scores by multiplying each value by number of occurences and then divide by total number, because of unbalanced classes!!

  # convert ndarray to list
  fpr = {k:v.tolist() for k, v in fpr.items()}
  tpr = {k:v.tolist() for k, v in tpr.items()}
  roc_auc = {k:v.tolist() for k, v in roc_auc.items()}

  # save all data
  utilities.save_json_file("roc_auc", roc_auc)
  utilities.save_json_file("fpr", fpr)
  utilities.save_json_file("tpr", tpr)