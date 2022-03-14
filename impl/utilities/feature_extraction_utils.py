import imp
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy
import collections
from scripts.svmModel import SVMModel
from utilities import general_utils


def extract_features(model, path):
  print("Extract features from...")
  category = ""
  features = {}
  for cat_name, file_name in general_utils.file_iterable(path):
    feature = model.extract(F'{path}{cat_name}/{file_name}')
    if category == cat_name:
      features[cat_name] = torch.cat([features[cat_name], feature.reshape(1, -1)], dim=0) # construct feature matrix
    else:
      print(F'Category: {cat_name}')
      features[cat_name] = feature.reshape(1, -1)
    category = cat_name
  return features


def extract_features_with_data_loader(model, train_loader):
  print("Extract features from...")
  category = ""
  features = {}
  for data, targets in train_loader: # iterate over training data
    feature = model.extract_from_loader(data)
    if category == targets[0]:
      features[targets[0]] = torch.cat([features[targets[0]], feature.reshape(1, -1)], dim=0) # construct feature matrix
    else:
      print(F'Category: {targets[0]}')
      features[targets[0]] = feature.reshape(1, -1)
    category = targets[0]
  return features


def save_plot(res):
    plt.figure()
    plt.plot(list(res.keys()), [obj["total_acc"] for obj in res.values()])
    plt.xlabel('Training Size') 
    plt.ylabel('Accuracy') 
    plt.savefig('total_acc.jpg')


def img_transforms():
  return torchvision.transforms.Compose([
    torchvision.transforms.Resize(224), # otherwise we would loose image information at the border
    torchvision.transforms.CenterCrop(224), # take only center from image
    torchvision.transforms.ToTensor(), # image to tensor
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),  # scale pixel values to range [-3,3]
    lambda x : x.unsqueeze(0) # required by pytorch (add batch dimension)
  ])


def prepare(model, path):
  print("Prepare data...")
  X_train = []
  y_train = []
  for cat_name, file_name in general_utils.file_iterable(path):
    feature = model.extract(F'{path}{cat_name}/{file_name}')
    X_train.append(feature)
    y_train.append(cat_name)
  X_train_std = model.fit_scaler(numpy.asarray(X_train)) # save transformation for test data and transform training data
  return (X_train_std, y_train)


def class_acc(data, category):
  print(F"Classified {data[0]:2} of {data[1]} images ({100 * data[0] / data[1]:6.2f}%) with {category}")
  return data[0] / data[1]


def total_acc(categories, class_accs, params):
  if params.unbalanced:
    acc_mean = numpy.mean(class_accs)
    print(F"\nAccuracy: {acc_mean * 100:6.2f}%")
    return acc_mean
  total = numpy.sum(list(categories.values()), axis=0)
  print(F"\nAccuracy: {total[0]:3} of {total[1]} images ({100 * total[0] / total[1]:6.2f}%)")
  return total[0] / total[1]


def predict(model, params, features=[]):
  print("Scoring...")
  categories = collections.defaultdict(lambda: [0,0]) # store number of correct identifictions and total number of identifications per category
  category = ""
  distances = []
  labels = []
  res = {}
  res["cat_acc"] = []
  res["categories"] = []
 
  if params.neighbor:
    max_number_features = max([len(f) for f in features.values()]) # variable with maximum number of features for one category
    distances = numpy.zeros((len(features), max_number_features)) # preserve all distances here

  if params.svm:
    svmModel = SVMModel(device="not used here")
    y_train = []
    X_train = []
    for key, val in features.items():
      y_train.extend(numpy.repeat(key, val.size()[0]))
      tmp = numpy.split(val.numpy(), val.size()[0])
      X_train.extend([i.flatten() for i in tmp])
    X_strain_std = svmModel.fit_scaler(X_train)
    svmModel.fit(X_strain_std, y_train)

  for cat_name, file_name in general_utils.file_iterable(params.d_test):
    X_test = model.extract(F'{params.d_test}{cat_name}/{file_name}')

    if params.svm:
      y_test = svmModel.predict(X_test)
    else:
      y_test = model.predict(X_test, features, distances, labels, params)

    categories[cat_name][0] += y_test == cat_name # we only increase when category has been correctly identified
    categories[cat_name][1] += 1 # always increase after each iteration s.t. we have the total number

    if category != cat_name and category:
      # print rates for the current category
      res["cat_acc"].append(class_acc(categories[category], category))
      res["categories"].append(category)
    category = cat_name

  res["cat_acc"].append(class_acc(categories[category], category)) # print last category
  res["categories"].append(category)
  res["total_acc"] = total_acc(categories, res["cat_acc"], params) # print total accuracy
  
  return res