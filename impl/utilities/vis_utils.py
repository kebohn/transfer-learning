#!/usr/bin/env python3

import argparse
import torch
import numpy
import matplotlib.pyplot as plt
from utilities import general_utils

def parse_arguments():
  parser = argparse.ArgumentParser(description='Visualizes features map and filters of a provided model')
  parser.add_argument('--model', type=general_utils.dir_path, help='Directory where model for visualization is located (absolute dir)')
  parser.add_argument('--d', type=general_utils.dir_path, help='Directory where images are stored that will be used for the visualization of the feature maps (absolute dir)')
  return parser.parse_args()


def visualize_filters(layers):
  for i, l in enumerate(layers):
    w = l.weight
    size = int(numpy.sqrt(w.shape[0])) # construct number of rows and columns for subplot
    x = y = int(size) + 1 if size % 1 != 0 else int(size) # check if number of filters can be arranged in subplots
    plt.figure(figsize=(20, 17))
    for j, filter in enumerate(w):
      plt.subplot(x, y, j+1) # use shape of filter to define subplot
      plt.imshow(filter[0, :, :].detach(), cmap='viridis') 
      plt.axis('off')
      plt.savefig(F'Conv_{i}_Filter.png')


def extractConvLayers(modules):
  layers = []
  for m in modules:
    if type(m) == torch.nn.Conv2d:
      layers.append(m)
    elif type(m) == torch.nn.Sequential:
      for i in m.children:
        if type(m) == torch.nn.Conv2d:
          layers.append(m)
  return layers


def get_feature_maps(input, layers):
  res = [layers[0](input)] # first output
  # consequently use the previous output as input for the next layer
  for l in range(1, len(layers)):
    res.append(layers[l](res[-1]))
  return res


def main():
  parsed_args = parse_arguments()
  
  # visualize convolutional layers with input images
  layers = extractConvLayers(parsed_args.model)
  visualize_filters(layers)

  input = 0 # TODO provide image for feature map
  get_feature_maps(input, layers)
  # TODO visualize feature map

if __name__ == "__main__":
  main()
  