#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import torch
import torchvision
import argparse
import models
import utilities


def parse_arguments():
  parser = argparse.ArgumentParser(description='Baseline script for transfer learning')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--d_test', type=utilities.dir_path, help='Directory where test files are stored (absolute dir)')
  parser.add_argument('--features', type=utilities.dir_path, help='Directory where features are stored (absolute dir)')
  parser.add_argument('--results', type=utilities.dir_path, help='Directory where results should be stored (absolute dir)')
  parser.add_argument('--cosine', dest='cosine', action='store_true', help='Apply cosine distance metric')
  parser.add_argument('--mean', dest='mean', action='store_true', help='Apply cosine distance on mean feature')
  parser.add_argument('--neighbor', dest='neighbor', action='store_true', help='Apply kNN metric')
  parser.add_argument('--svm', dest='svm', action='store_true', help='Apply Support Vector Machine')
  parser.add_argument('--k', type=int, dest='k', default=5, help='Define k for kNN algorithm (Default: k=5)')
  parser.add_argument('--step', type=int, dest='step', default=5, help='Define step with which training set should be decreased (Default: k=5)')
  parser.add_argument('--max-size', type=int, dest='max_size', default=5, help='Define maximum samples per class (Default: k=5)')
  parser.add_argument('--unbalanced', dest='unbalanced', action='store_true', help='Define if dataset is unbalanced (Default: false)')
  return parser.parse_args()

def main():
  parsed_args = parse_arguments()
  res = {}
      
  # use Feature Extraction Model
  res50_model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
  model = models.FEModel(model=res50_model, device=utilities.get_device())

  # define current training size per category
  current_size = parsed_args.step

  # load test data
  test_data = utilities.CustomImageDataset('data.csv', parsed_args.d_test, utilities.test_transforms())
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False) 

  # increase current size per category by step_size after every loop
  while(current_size <= parsed_args.max_size):
    print(F'Using {current_size} images per category...')
    
    # load existing features
    if parsed_args.features:
      features = torch.load(F'{parsed_args.features}features_size_{current_size}.pt')

    # compute new features
    else:
      # load data
      train_data = utilities.CustomImageDataset('data.csv', parsed_args.d, utilities.train_transforms(), current_size)
      train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=False, num_workers=5)

      # extract features from training data
      features = utilities.extract(model, train_loader)
      torch.save(features, F'{parsed_args.results}features_size_{current_size}.pt')
  
    res[current_size] = utilities.predict(
        model=model,
        params=parsed_args,
        features=features,
        test_loader=test_loader
      )
    current_size += parsed_args.step
   
  # write result to a file
  utilities.save_json_file(F'{parsed_args.results}res', res) 
  utilities.save_training_size_plot(parsed_args.results, res)


if __name__ == "__main__":
  main()
  