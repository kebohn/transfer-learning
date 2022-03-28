#!/usr/bin/env python3
import sys

sys.path.append("..") # append the path of the parent directory

import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import utilities
import models


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
  plt.title("Receiver operating characteristic example")
  plt.legend(loc="lower right")
  plt.savefig(name)
  plt.close()


def perform_roc(method, features, features_test):
      y_test = []
      for ctn, (_, val) in enumerate(features_test.items()):
        key_list = numpy.repeat(ctn, val.size(0)) # add the same amount of category label to list as samples
        y_test.extend(key_list) # add the current category label list to whole label list

      # one hot encoding of test labels
      y = label_binarize(y_test, classes=list(set(y_test)))

      if method == 'svm':
      

        model = models.FEModel(None, utilities.get_device())
        svmModel = models.SVMModel(device="not used here", probability=True)
        y_train = []
        X_train = []
        features_norm = model.normalize_train(features)
        for key, val in features_norm.items():
          y_train.extend(numpy.repeat(key, val.size()[0]))
          tmp = numpy.split(val.cpu().numpy(), val.size()[0])
          X_train.extend([i.flatten() for i in tmp])

        svmModel.fit(X_train, y_train)

        # normalize test features 
        features_test_norm = model.normalize_test(features_test)

        # combine features into one tensor
        features_test_tensor = torch.cat(tuple(features_test_norm.values()), dim=0)

        # get desicion function with test features
        scores = svmModel.model.predict_proba(features_test_tensor)


      # plot roc curve
      fpr = dict()
      tpr = dict()
      roc_auc = dict()

      # iterate over each class
      for i in range(y.shape[1]):

        # compute false positive rate and true positive rate for each class
        fpr[i], tpr[i], _ = roc_curve(y[:,i], scores[:, i])
 

        # compute area under the curve for each class
        roc_auc[i] = auc(fpr[i], tpr[i])

        save_roc_curve(F'roc_class_{i}.png', fpr[i], tpr[i], roc_auc[i])