#!/usr/bin/env python3

import torch
import numpy
import os, argparse, collections
from dlModel import DLModel
from svmModel import SVMModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
  parser = argparse.ArgumentParser(description='Baseline script for transfer learning')
  parser.add_argument('--d', type=dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--d_test', type=dir_path, help='Directory where test files are stored (absolute dir)')
  parser.add_argument('--extract', dest='extract', action='store_true', help='Extract features and store it')
  parser.add_argument('--cosine', dest='cosine', action='store_true', help='Apply cosine distance metric')
  parser.add_argument('--mean', dest='mean', action='store_true', help='Apply cosine distance on mean feature')
  parser.add_argument('--neighbor', dest='neighbor', action='store_true', help='Apply kNN metric')
  parser.add_argument('--svm', dest='svm', action='store_true', help='Apply Support Vector Machine')
  parser.add_argument('--k', type=int, dest='k', default=5, help='Define k for kNN algorithm (Default: k=5)')
  parser.add_argument('--step', type=int, dest='step', default=5, help='Define step with which training set should be decreased (Default: k=5)')
  return parser.parse_args()


def dir_path(path):
  if os.path.isabs(path):
    return path
  raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")


def file_iterable(path):
  for cat_dir in sorted(os.listdir(path)):
    cat_name = os.fsdecode(cat_dir)
    for file_name in sorted(os.listdir(F'{path}{cat_name}')):
      yield (cat_name, file_name)


def extract_features(model, path):
  print("Extract features from...")
  category = ""
  features = {}
  for cat_name, file_name in file_iterable(path):
    feature = model.extract(F'{path}{cat_name}/{file_name}')
    if category == cat_name:
      features[cat_name] = torch.cat([features[cat_name], feature.reshape(1, -1)], dim=0) # construct feature matrix
    else:
      print(F'Category: {cat_name}')
      features[cat_name] = feature.reshape(1, -1)
    category = cat_name
  return features


def prepare(model, path):
  print("Prepare data...")
  X_train = []
  y_train = []
  for cat_name, file_name in file_iterable(path):
    feature = model.extract(F'{path}{cat_name}/{file_name}')
    X_train.append(feature)
    y_train.append(cat_name)
  X_train_std = model.fit_scaler(numpy.asarray(X_train)) # save transformation for test data and transform training data
  return (X_train_std, y_train)


def print_class_acc(data, category):
  print(F"Classified {data[0]:2} of {data[1]} images ({100 * data[0] / data[1]:6.2f}%) with {category}")


def print_total_acc(categories):
  total = numpy.sum(list(categories.values()), axis=0)
  print(F"\nAccuracy: {total[0]:3} of {total[1]} images ({100 * total[0] / total[1]:6.2f}%)")


def predict(model, params, features=[]):
  print("Scoring...")
  categories = collections.defaultdict(lambda: [0,0]) # store number of correct identifiactions and total number of identifications per category
  category = ""
  distances = []
  labels = []

  if params.neighbor:
    max_number_features = max([len(f) for f in features.values()]) # variable with maximum number of features for one category
    distances = numpy.zeros((len(features), max_number_features)) # preserve all distances here

  for cat_name, file_name in file_iterable(params.d_test):
    X_test = model.extract(F'{params.d_test}{cat_name}/{file_name}')
    y_test = model.predict(X_test, features, distances, labels, params)

    categories[cat_name][0] += y_test == cat_name # we only increase when category has been correctly identified
    categories[cat_name][1] += 1 # always increase after each iteration s.t. we have the total number

    if category != cat_name and category:
      # print rates for the current category
      print_class_acc(categories[category], category)
    category = cat_name

  print_class_acc(categories[category], category) # print last category
  print_total_acc(categories) # print total accuracy


def main():
  parsed_args = parse_arguments()
  if (parsed_args.svm): # use SVM Model
    model = SVMModel(device=device)
    X_train, y_train = prepare(model, parsed_args.d)
    for X_train_filtered, y_train_filtered in model.step_iter(X_train, y_train, parsed_args.step):
      model.fit(X_train_filtered, y_train_filtered)
      predict(model=model, params=parsed_args)
  else: # use Deep Learning Model
    model = DLModel(device=device)
    if parsed_args.extract:
      features = extract_features(model, parsed_args.d)
      torch.save(features, 'features.pt')
    else:
      features = torch.load('features.pt')
    for features_filtered in model.step_iter(features, parsed_args.step):
      predict(model=model, params=parsed_args, features=features_filtered)


if __name__ == "__main__":
  main()
  