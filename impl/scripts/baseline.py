#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import torch
import torchvision
import argparse, json
from feModel import FEModel
import utilities

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
  parser = argparse.ArgumentParser(description='Baseline script for transfer learning')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--d_test', type=utilities.dir_path, help='Directory where test files are stored (absolute dir)')
  parser.add_argument('--extract', dest='extract', action='store_true', help='Extract features and store it')
  parser.add_argument('--cosine', dest='cosine', action='store_true', help='Apply cosine distance metric')
  parser.add_argument('--mean', dest='mean', action='store_true', help='Apply cosine distance on mean feature')
  parser.add_argument('--neighbor', dest='neighbor', action='store_true', help='Apply kNN metric')
  parser.add_argument('--svm', dest='svm', action='store_true', help='Apply Support Vector Machine')
  parser.add_argument('--k', type=int, dest='k', default=5, help='Define k for kNN algorithm (Default: k=5)')
  parser.add_argument('--step', type=int, dest='step', default=5, help='Define step with which training set should be decreased (Default: k=5)')
  parser.add_argument('--unbalanced', dest='unbalanced', action='store_true', help='Define if dataset is unbalanced (Default: false)')
  return parser.parse_args()

def main():
  parsed_args = parse_arguments()
  res = {}
      
  # use Feature Extraction Model
  res50_model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
  model = FEModel(model=res50_model, transforms=utilities.img_transforms(), device=device)
  if parsed_args.extract:
    features = utilities.extract_features(model, parsed_args.d)
    torch.save(features, 'features.pt')
  else:
    features = torch.load('features.pt')
  for features_filtered, n in model.step_iter(features, parsed_args.step):
    res[n] = utilities.predict(
        model=model,
        params=parsed_args,
        features=features_filtered
      )
   
  # write result to a file
  with open('res.json', 'w') as fp:
    json.dump(res, fp,  indent=4)
      
  utilities.save_plot(res)


if __name__ == "__main__":
  main()
  