from utilities.cuda import get_device
import torch
import matplotlib.pyplot as plt
import numpy
import collections
import models
import data


def softmax(x):
  return numpy.exp(x) / sum(numpy.exp(x))


def prepare_features_for_training(pre_trained_model, train_loader, features_valid):
 
  # extract training features from training data
  features_train = extract(pre_trained_model, train_loader)

  # normalize training features 
  features_train_norm = pre_trained_model.normalize_train(features_train)

  # handle trainig features like a dataset
  feature_train_data = data.FeatureDataset(features_train_norm)
  feature_train_loader = torch.utils.data.DataLoader(dataset=feature_train_data, batch_size=10, shuffle=True)

  # normalize validation features according train normalization
  features_valid_norm = pre_trained_model.normalize_test(features_valid)

  # handle validation features like a dataset
  feature_valid_data = data.FeatureDataset(features_valid_norm)
  feature_valid_loader = torch.utils.data.DataLoader(dataset=feature_valid_data, batch_size=10, shuffle=False)

  return feature_train_loader, feature_valid_loader

 
def extract(model, train_loader):
  print("Extract features...")
  res = {}

  # iterate over training data
  for data, _, names in train_loader:
    features = model.extract(data) # extract features for whole batch
    cat_set = set(names)
    names_arr = numpy.array(names)
    for category in cat_set: # iterate over all distinctive categories in the batch
      indices = numpy.argwhere(names_arr == category).flatten() # find indices from the same category
      cat_features = torch.index_select(features, 0, torch.from_numpy(indices).to(get_device())) # retrieve only features from correct category
      cat_features = cat_features.unsqueeze(dim=0) if len(cat_features.size()) == 1 else cat_features # make sure we have a 2D tensor
      if category in res.keys(): # check if we already have some features
        res[category] = torch.cat((res[category], cat_features), dim=0) # add new features to existing ones
      else:
        res[category] = cat_features # add new features
  return dict(sorted(res.items()))


def save_training_size_plot(res_dir, res):
  plt.figure()
  plt.plot(list(res.keys()), [obj["total_acc"] for obj in res.values()])
  plt.xlabel('Training Size') 
  plt.ylabel('Accuracy') 
  plt.savefig(F'{res_dir}total_acc.jpg')
  plt.close()


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


def predict(model, params, features=[], test_loader=[]):
  print("Scoring...")
  categories = collections.defaultdict(lambda: [0,0]) # store number of correct identifictions and total number of identifications per category
  category = ""
  distances = []
  labels = []
  res = {}
  res["cat_acc"] = []
  res["categories"] = []
  res["labels"] = []
  res["predictions"] = []
 
  if params.neighbor:
    max_number_features = max([len(f) for f in features.values()]) # variable with maximum number of features for one category
    distances = numpy.zeros((len(features), max_number_features)) # preserve all distances here

  if params.svm:
    svmModel = models.SVMModel(device="not used here")
    y_train = []
    X_train = []
    features_norm = model.normalize_train(features)
    for key, val in features_norm.items():
      y_train.extend(numpy.repeat(key, val.size()[0]))
      tmp = numpy.split(val.detach().cpu().numpy(), val.size()[0])
      X_train.extend([i.flatten() for i in tmp])

    svmModel.fit(X_train, y_train)

  for test_data, _, test_name in test_loader:

    # convert tuple to string
    test_name = ''.join(test_name)

    # add test label to res array
    res["labels"].append(test_name)

    # extract test feature from model
    X_test = model.extract(test_data)

    if params.svm:
      X_test_norm = model.normalize(X_test) # normalize test data with norm from training data
      y_test = svmModel.predict(X_test_norm.detach().cpu().reshape(1, -1))
    else:
      y_test, _ = model.predict(X_test, features, distances, labels, params)

    # add test prediction to res array
    res["predictions"].append(y_test) 

    categories[test_name][0] += y_test == test_name # we only increase when category has been correctly identified
    categories[test_name][1] += 1 # always increase after each iteration s.t. we have the total number

    if category != test_name and category:
      # print rates for the current category
      res["cat_acc"].append(class_acc(categories[category], category))
      res["categories"].append(category)
    category = test_name

  res["cat_acc"].append(class_acc(categories[category], category)) # print last category
  res["categories"].append(category)
  res["total_acc"] = total_acc(categories, res["cat_acc"], params) # print total accuracy
  
  return res