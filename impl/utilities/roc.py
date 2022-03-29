#!/usr/bin/env python3
import sys

sys.path.append("..") # append the path of the parent directory

import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
  plt.title("ROC Indoor")
  plt.legend(loc="lower right")
  plt.savefig(name)
  plt.close()


def perform_roc(method, features, features_test, model=None):

      # prepare test labels
      y_test = []
      for ctn, (_, val) in enumerate(features_test.items()):
        key_list = numpy.repeat(ctn, val.size(0)) # add the same amount of category label to list as samples
        y_test.extend(key_list) # add the current category label list to whole label list

      # one hot encoding of test labels
      y = label_binarize(y_test, classes=list(set(y_test)))

      # define normalization
      feModel = models.FEModel(None, utilities.get_device())
      features_norm = feModel.normalize_train(features)
      features_test_norm = feModel.normalize_test(features_test)

      if method == "svm":
        svmModel = models.SVMModel(device="not used here", probability=True)

        # prepare train data
        y_train = []
        X_train = []
        for key, val in features_norm.items():
          y_train.extend(numpy.repeat(key, val.size()[0]))
          tmp = numpy.split(val.cpu().numpy(), val.size()[0])
          X_train.extend([i.flatten() for i in tmp])

        svmModel.fit(X_train, y_train)

        # combine features into one tensor
        features_test_tensor = torch.cat(tuple(features_test_norm.values()), dim=0)

        # get decision function with test features
        scores = svmModel.model.predict_proba(features_test_tensor)

      elif method == "cosine":

        # combine features into one tensor
        features_tensor = torch.cat(tuple(features.values()), dim=0)
        features_test_tensor = torch.cat(tuple(features_test.values()), dim=0)

        # compute cosine similarity
        sim = cosine_similarity(features_test_tensor.cpu(), features_tensor.cpu())

        # create probabilty score matrix for each sample
        sim_proba = numpy.zeros(y.shape)

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
        scores = numpy.apply_along_axis(utilities.softmax, 1, sim_proba)

      elif method == "mean":

        # build mean feature for all classes
        features_mean = dict()
        for key, vals in features.items():
          features_mean[key] = torch.unsqueeze(torch.mean(vals, 0), 0)

        # combine features into one tensor
        features_mean_tensor = torch.cat(tuple(features_mean.values()), dim=0)
        features_test_tensor = torch.cat(tuple(features_test.values()), dim=0)

        # compute cosine similarity
        sim = cosine_similarity(features_test_tensor.cpu(), features_mean_tensor.cpu())

        # apply softmax on each row such that we have a probability for each class that sums up to 1
        scores = numpy.apply_along_axis(utilities.softmax, 1, sim)

      # following methods will use direct classification with logits and softmax
      elif method == "pretrained":

        # here the features are simply the test set
        test_loader = torch.utils.data.DataLoader(dataset=features_test, batch_size=y.shape[0], shuffle=False)

        for data, targets, _ in test_loader: # iterate over test data in batches - her we take everything at once
          data = data.to(utilities.get_device())
          targets = targets.to(utilities.get_device())

          # forward pass
          scores = model(data)

      elif method == "adaptive":
        # handle test features like a dataset
        feature_test_data = data.FeatureDataset(features_test_norm)
        feature_test_loader = torch.utils.data.DataLoader(dataset=feature_test_data, batch_size=y.shape[0], shuffle=False)

        for data, targets, _ in feature_test_loader: # iterate over test data in batches - her we take everything at once
          data = data.to(utilities.get_device())
          targets = targets.to(utilities.get_device())

          # forward pass
          scores = model(data)

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

      # First aggregate all false positive rates
      all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(y.shape[1])]))

      # Then interpolate all ROC curves at this points
      mean_tpr = numpy.zeros_like(all_fpr)
      for i in range(y.shape[1]):
          mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])

      # Finally average it and compute AUC
      mean_tpr /= y.shape[1]

      makro_auc = auc(all_fpr, mean_tpr)

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
