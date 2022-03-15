#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import argparse
import torch
import numpy
import matplotlib.pyplot as plt
import utilities

vis = {} # dict stores all layer outputs

def parse_arguments():
  parser = argparse.ArgumentParser(description='Visualizes features map and filters of a provided model')
  parser.add_argument('--model', type=utilities.dir_path, help='Directory where model for visualization is located (absolute dir)')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where images are stored that will be used for the visualization of the feature maps (absolute dir)')
  parser.add_argument('--fine-tune', dest='fine_tune', action='store_true', help='Define if the whole model is fine-tuned (Default: false)')
  parser.add_argument('--filters', dest='filters', action='store_true', help='Visualize filters of model (Default: false)')
  parser.add_argument('--maps', dest='maps', action='store_true', help='Visualize feature maps of model (Default: false)')
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
    plt.close()


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



def hook_fn(m, i, o):
  print(m)
  print(i)
  vis[m] = o 

def get_layers(model):
  for _, layer in model._modules.items():
    # recursive call on children of sequential object
    if isinstance(layer, torch.nn.Sequential):
      get_layers(layer)
    else:
      # only register a hook when we do not have sequential object
      layer.register_forward_hook(hook_fn)

def main():
  parsed_args = parse_arguments()

  # load test data
  test_data = utilities.CustomImageDataset('data.csv', parsed_args.d, utilities.test_transforms())
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False) 

  # define and load existing model
  model = utilities.define_model(test_data, parsed_args.fine_tune)
  model.load_state_dict(torch.load(parsed_args.model))
  model.eval()

  # visualize convolutional layers with input images
  if parsed_args.filters:
    layers = extractConvLayers(parsed_args.model)
    visualize_filters(layers)

  if parsed_args.maps:
    get_layers(model)
    for img, label, name in test_loader:
      print(F"Save Feature maps for category {label} -> {name[0]}")
      img.to(utilities.get_device())
      res = model(img)
      print(F"Classified: {res}")
      break
    utilities.save_json_file("teest", vis)


if __name__ == "__main__":
  main()
  