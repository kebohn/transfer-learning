#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import torch
import torchvision
import argparse
import utilities
import models
import data


def parse_arguments():
  parser = argparse.ArgumentParser(description='Baseline script for transfer learning')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--d_test', type=utilities.dir_path, help='Directory where test files are stored (absolute dir)')
  parser.add_argument('--model', type=utilities.dir_path, help='Directory where model is stored (absolute dir)')
  parser.add_argument('--features', type=utilities.dir_path, help='Directory where features are stored (absolute dir)')
  parser.add_argument('--results', type=utilities.dir_path, help='Directory where results should be stored (absolute dir)')
  parser.add_argument('--extract', dest='extract', action='store_true', help='Extract features and store it')
  parser.add_argument('--cosine', dest='cosine', action='store_true', help='Apply cosine distance metric')
  parser.add_argument('--mean', dest='mean', action='store_true', help='Apply cosine distance on mean feature')
  parser.add_argument('--neighbor', dest='neighbor', action='store_true', help='Apply kNN metric')
  parser.add_argument('--svm', dest='svm', action='store_true', help='Apply Support Vector Machine')
  parser.add_argument('--k', type=int, dest='k', default=5, help='Define k for kNN algorithm (Default: k=5)')
  parser.add_argument('--step', type=int, dest='step', default=5, help='Define step with which training set should be decreased (Default: k=5)')
  parser.add_argument('--max-size', type=int, dest='max_size', default=5, help='Define maximum samples per class (Default: k=5)')
  parser.add_argument('--unbalanced', dest='unbalanced', action='store_true', help='Define if dataset is unbalanced (Default: false)')
  parser.add_argument('--early-stop', dest='early_stop', action='store_true', help='Define if training should be stopped when plateau is reached (Default: false)')
  parser.add_argument('--auc', dest='auc', action='store_true', help='Area under the curve scheme for early stopping, if false validation loss will be used (Default: false)')
  return parser.parse_args()


def main():
  parsed_args = parse_arguments()

  # hyperparameters
  epochs = 100
  lr = 0.01
  momentum = 0.9
  current_size = parsed_args.step
  res = {}

  # load pre-trained model for feature extraction
  res50_model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
  pre_trained_model = models.FEModel(model=res50_model, device=utilities.get_device())

  # load test and validation data
  print("Prepare test and validation dataset...")
  valid_data = data.CustomImageDataset('validation.csv', parsed_args.d, utilities.test_transforms())
  test_data = data.CustomImageDataset('data.csv', parsed_args.d_test, utilities.test_transforms())

  valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=10, shuffle=False, num_workers=8)
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=10, shuffle=False, num_workers=8)

  # extract test and validation data
  features_valid = utilities.extract(pre_trained_model, valid_loader)
  features_test = utilities.extract(pre_trained_model, test_loader)

  # increase current size per category by step_size after every loop
  while(current_size <= parsed_args.max_size):
    print(F'Using {current_size} images per category...')

    # define adapter model - must be always 
    adapter_model = models.AdaptiveModel(num_categories=test_data.get_categories())
    adapter_model.to(utilities.get_device()) # save to GPU

    # load training data
    train_data = data.CustomImageDataset('data.csv', parsed_args.d, utilities.train_transforms(), current_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=False, num_workers=8)

    # train data with current size of samples per category
    f_train_loader, _ = utilities.train(
      pre_trained_model=pre_trained_model,
      adapter_model=adapter_model, 
      epochs=epochs, 
      lr=lr,
      momentum=momentum, 
      train_loader=train_loader,
      features_valid=features_valid,
      parsed_args=parsed_args,
      current_size=current_size
    )

    # normalize test data 
    features_test_norm = pre_trained_model.normalize_test(features_test)

    # handle test features like a dataset
    feature_test_data = data.FeatureDataset(features_test_norm)
    feature_test_loader = torch.utils.data.DataLoader(dataset=feature_test_data, batch_size=1, shuffle=False)

    # extract features from model and use this with another specified metric to predict the categories
    if parsed_args.extract:

      # use Feature Extraction Model
      adaptive_features_model = models.FEModel(model=adapter_model, device=utilities.get_device())
        
      # extract features from trained adaptive model
      adaptive_train_features = utilities.extract(adaptive_features_model, f_train_loader)
      adaptive_test_features = utilities.extract(adaptive_features_model, feature_test_loader)

      # save features
      torch.save(adaptive_train_features, F'{parsed_args.results}features_train_size_{current_size}.pt')
      torch.save(adaptive_test_features, F'{parsed_args.results}features_test_size_{current_size}.pt')

      # run prediction
      res[current_size] = utilities.predict(
          model=adaptive_features_model,
          params=parsed_args,
          features=adaptive_train_features,
          test_loader=feature_test_loader,
        )

    # use the model to classify the images
    else:
      res[current_size] = utilities.test(adapter_model, feature_test_loader)

    current_size += parsed_args.step

  utilities.save_json_file(F'{parsed_args.results}res', res)
  utilities.save_training_size_plot(parsed_args.results, res)


if __name__ == "__main__":
  main()