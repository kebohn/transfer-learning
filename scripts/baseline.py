#!/usr/bin/env python3

import torch
import os, argparse
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
  parser.add_argument('--n', type=int, dest='n', default=5, help='Define amount of used training samples (Default: k=5)')
  return parser.parse_args()


def dir_path(path):
  if os.path.isabs(path):
      return path
  else:
      raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")


def extract_features(args, model):
  print("Extracting features from...")
  features = {}
  for cat_dir in sorted(os.listdir(args.d)):
    print(F"category: {cat_dir}")
    cat_name = os.fsdecode(cat_dir)
    files = sorted(os.listdir(F'{args.d}{cat_name}'))
    idx = 0
    features[cat_name] = torch.empty((len(files), 2048)) # 2048 is number of elements in last layer of ResNet-50
    for file in files:
      feature = model.extract(F'{args.d}{cat_name}/{file}')
      features[cat_name][idx,:] = feature
      idx += 1

  return features


def main():
  parsed_args = parse_arguments()
  if (parsed_args.svm):
    model = SVMModel(device=device, params=parsed_args)
  else: 
    model = DLModel(device=device, params=parsed_args)
  if parsed_args.extract:
    features = extract_features(parsed_args, model)
    torch.save(features, 'features.pt')
  else: 
    features = torch.load('features.pt')
  model.predict(features)


if __name__ == "__main__":
  main()


