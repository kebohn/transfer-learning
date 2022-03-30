#!/usr/bin/env python3
import dataclasses
import sys

sys.path.append("..") # append the path of the parent directory

import argparse
import torch
import torchvision
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import utilities
import data
import models

vis = {} # dict stores all layer outputs

def parse_arguments():
  parser = argparse.ArgumentParser(description='Visualizes features map and filters of a provided model')
  parser.add_argument('--model', type=utilities.dir_path, help='Directory where model for visualization is located (absolute dir)')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where images are stored that will be used for the visualization of the feature maps (absolute dir)')
  parser.add_argument('--fine-tune', dest='fine_tune', action='store_true', help='Define if the whole model is fine-tuned (Default: false)')
  parser.add_argument('--filters', dest='filters', action='store_true', help='Visualize filters of model (Default: false)')
  parser.add_argument('--maps', dest='maps', action='store_true', help='Visualize feature maps of model (Default: false)')
  parser.add_argument('--dm', dest='dm', action='store_true', help='Apply dimensionality reduction (t-sne, pca) on features (Default: false)')
  parser.add_argument('--roc', dest='roc', action='store_true', help='Visualize RoC graph on features (Default: false)')
  parser.add_argument('--features', type=utilities.dir_path, help='Directory where features are stored (absolute dir)')
  parser.add_argument('--features_test', type=utilities.dir_path, help='Directory where test features are stored (absolute dir)')
  parser.add_argument('--d_test', type=utilities.dir_path, help='Directory where test data are stored (absolute dir)')
  parser.add_argument('--confusion', type=utilities.dir_path, help='Directory where data for confusion matrix is stored (absolute dir)')
  return parser.parse_args()


def visualize_filters(layers):
  for i, l in enumerate(layers):
    w = l.weight
    size = int(numpy.sqrt(w.shape[0])) # construct number of rows and columns for subplot
    x = y = int(size) + 1 if size % 1 != 0 else int(size) # check if number of filters can be arranged in subplots
    plt.figure(figsize=(20, 17))
    for j, filter in enumerate(w):
      plt.subplot(x, y, j+1) # use shape of filter to define subplot
      plt.imshow(filter[0, :, :].cpu().detach(), cmap='viridis') 
      plt.axis('off')
      plt.savefig(F'Conv_{i}_Filter.png')
    plt.close()


def extractConvLayers(model):
  modules = list(model.children())
  layers = []
  for m in modules:
    if type(m) == torch.nn.Conv2d:
      layers.append(m)
    elif type(m) == torch.nn.Sequential:
      for i in list(m.children()):
        if type(i) == torch.nn.Conv2d:
          layers.append(i)
  return layers


def hook_fn(module, _, output):
  vis[module] = output 


def get_layers(model):
  for _, layer in model._modules.items():
    # recursive call on children of sequential object
    if isinstance(layer, torch.nn.Sequential):
      get_layers(layer)
    else:
      # only register a hook when we do not have a sequential object
      layer.register_forward_hook(hook_fn)


def save_scatter_plot(features, proj, num_categories, name):
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1,1,1)
    index_start = 0
    counter = 0
    for cat, val in features.items():
        index_end = index_start + val.size(0) - 1 # calculate end of category index
        proj_cat = proj[index_start:index_end, :] # extract only values for one category
        ax.scatter(proj_cat[:, 0], proj_cat[:, 1], c=utilities.colors[counter], label = cat, alpha=0.5, marker='+')
        index_start = index_end + 1
        counter += 1
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]) # Shrink current axis's height by 10% on the bottom
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=int(num_categories / 8))
    plt.savefig(F'{name}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def normalize_features(features):
  vals = torch.cat(tuple(features.values()), dim=0) # combine features into one tensor
  norm = torch.linalg.norm(vals,dim=0) # compute norm over the columns
  tol = 1e-12 # tolerance

  # check if computed norm is greater than tolerance to prevent divsion by zero
  norm[norm < tol] = tol

  # apply normalization
  return {k: normalize(v, norm) for k, v in features.items()}


def normalize(features, norm):
  # use same norm from training features
  return torch.div(features, norm)


def main():
  parsed_args = parse_arguments()

  if parsed_args.features is not None:
    features = torch.load(parsed_args.features)
    if parsed_args.dm:
      
      f_dim = (list(features.values())[0]).size(1) # get feature dimension (retrieved from first element)
      num_categories = len(features.keys()) # get number of categories
      embeddings = torch.zeros((0, f_dim), dtype=torch.float32) # init embedding with calculated f dimension
      for f in features.values():
        # stack all features from every class into one embeddings tensor
        embeddings = torch.cat((embeddings, f))

      # invoke t-SNE on stacked feature embedding
      tsne = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto', verbose=1)
      tsne_proj = tsne.fit_transform(embeddings)

      # visualize t-sne with coloring of correct class
      save_scatter_plot(features, tsne_proj, num_categories, 'tsne')

      # invoke pca algo
      pca = PCA(n_components=2)
      pca_proj = pca.fit_transform(embeddings)
      print(F'Variance ratio: {pca.explained_variance_ratio_}')

      # visualize pca with coloring of correct class
      save_scatter_plot(features, pca_proj, num_categories, 'pca')
      
    if parsed_args.roc:
      model = None
      features_test = torch.load(parsed_args.features_test)
      if parsed_args.d_test is not None:
        # treat inital image data as features
        features_test = test_data = data.CustomImageDataset('data.csv', parsed_args.d_test, utilities.test_transforms())
        model = torchvision.models.resnet50(pretrained=True)
      
      model = models.AdaptiveModel(num_categories=y.shape[1])
      model.load_state_dict(torch.load(model))
      model.eval()
      utilities.perform_roc("pretrained", features, features_test, model)

  if parsed_args.confusion is not None:
      
    res_data = utilities.load_json_file(parsed_args.confusion)

    # only test on 70 images per class at the moment
    res_data = res_data["5"]

    print(res_data["cat_acc"])

    # create confusion matrix
    labels = list(numpy.unique(numpy.array(res_data["labels"])))
    confusion = confusion_matrix(res_data["labels"], res_data["predictions"], labels=labels)

    # transpose matrix because we want the rows to be the predicted class
    confusion = confusion.T

    # append a row with rounded accuracy values
    acc_row = numpy.around(numpy.array(res_data["cat_acc"]) * 100, decimals=0)
    confusion = numpy.vstack([confusion, acc_row]).astype(int)
    y_labels = labels.copy()
    y_labels.append("%")

    fig, ax = plt.subplots(figsize=(20,20))

    # mask the accuracy / bottom row
    confusion_bottom = confusion.copy()
    masked_array = numpy.ones(confusion.shape, dtype=bool)
    false_list = numpy.zeros((1, confusion.shape[1]), dtype=bool)
    masked_array[-1,:] = false_list
    mask_confusion_bottom = numpy.ma.MaskedArray(confusion_bottom, mask=masked_array)
   
    # mask diagonal matrix
    confusion_diag = confusion.copy()
    numpy.fill_diagonal(confusion_diag, acc_row)
    masked_array = numpy.ones(confusion.shape, dtype=bool)
    false_list = numpy.zeros((1, confusion.shape[1]), dtype=bool)
    numpy.fill_diagonal(masked_array, false_list)
    mask_confusion_diag = numpy.ma.MaskedArray(confusion_diag, mask=masked_array)

    im = ax.imshow(confusion, cmap='jet', vmin=0, vmax=numpy.amax(confusion[:-1,:]))
    im_bottom = ax.imshow(confusion_bottom, cmap=mpl.colors.ListedColormap(['white']))
    im_diag = ax.imshow(confusion_diag, cmap='jet', vmin=0, vmax=numpy.amax(acc_row))
    
    im_bottom.set_data(mask_confusion_bottom)
    im_diag.set_data(mask_confusion_diag)

    print(labels)

    # show ticks and labels on both axes
    ax.set_xticks(numpy.arange(len(labels)))
    ax.set_yticks(numpy.arange(len(y_labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Estimated')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
      color = "w" if i != confusion.shape[0] - 1 else "black"
      for j in range(len(labels)):
          if confusion[i,j] != 0: # do not show 0 values
            ax.text(j, i, confusion[i, j], ha="center", va="center", color=color)

    ax.set_title("Confusion matrix Indoor")
    fig.tight_layout()
    plt.tick_params(bottom = False)
    plt.tick_params(left = False)
    plt.savefig("confusion_matrix.png")
    plt.close()

  if parsed_args.filters or parsed_args.maps:
    # load test data
    test_data = data.CustomImageDataset('data.csv', parsed_args.d, utilities.test_transforms())
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False) 

    # define and load existing model
    model = utilities.define_model(test_data, parsed_args.fine_tune)
    model.load_state_dict(torch.load(parsed_args.model))
    model.eval()

    # visualize convolutional layers with input images
    if parsed_args.filters:
      layers = extractConvLayers(model)
      visualize_filters(layers)

    if parsed_args.maps:
      model.cpu()
      get_layers(model)
      with torch.no_grad():
        for img, label, name in test_loader:
          print(F"Save Feature maps for category {label} -> {name[0]}")
          _ = model(img)
          break
  
      for mod, output in vis.items():
        # remove batch dimension
        d = output.squeeze()

        size = int(numpy.sqrt(d.size(0))) # construct number of rows and columns for subplot
        x = y = int(size) + 1 if size % 1 != 0 else int(size) # check if number of filters can be arranged in subplots
        plt.figure(figsize=(20, 17))
        for j, activation in enumerate(d):
          plt.subplot(x, y, j+1) # use shape of filter to define subplot
          plt.imshow(activation, cmap='viridis') 
          plt.axis('off')
          plt.savefig(F'{mod._get_name()}_Activation.png')
        plt.close()
        break

if __name__ == "__main__":
  main()
  