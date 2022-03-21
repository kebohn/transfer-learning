#!/usr/bin/env python3
from random import sample
import sys
sys.path.append("..") # append the path of the parent directory

import argparse
import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import utilities

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

def save_roc_curve(name, fpr, tpr, roc_auc):
  plt.figure()
  lw = 2
  plt.plot(
      fpr[2],
      tpr[2],
      color="darkorange",
      lw=lw,
      label=F"ROC curve (area = {roc_auc[2]:.2f})",
  )
  plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver operating characteristic example")
  plt.legend(loc="lower right")
  plt.savefig(F"{name}.png")
  plt.close()
 

def main():
  parsed_args = parse_arguments()

  if parsed_args.features is not None:
    features = torch.load(parsed_args.features)
    if  parsed_args.dm:
      
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

      plt.figure()

      # iterate over all classes
      fpr = tpr = thresh = roc_auc = {}
      cat_list = list(features.keys())
      for idx, (cat, vals) in enumerate(features.items()):

        # compute pairwise similarity of all features in one class
        sim_dist_self = 1 - cosine_similarity(vals.cpu())

        # filter out the current class
        features_other = {k:v for k,v in features.items() if k != cat}

        # concat all other features in one tensor
        vals_other = torch.cat(tuple(features_other.values()), dim=0) # combine features into one tensor

        # compute pairwise similarity with all other classes
        sim_dist_other = 1 - cosine_similarity(vals.cpu(), vals_other.cpu())

        # perfom roc scheme
        fpr[idx], tpr[idx], thresh[idx] = roc_curve(sim_dist_self, sim_dist_other)
        roc_auc[idx] = auc(fpr[idx], tpr[idx])
        # plotting    
        plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')

        break
      plt.title('Multiclass ROC curve')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive rate')
      plt.legend(loc='best')
      plt.savefig('Multiclass ROC',dpi=300);  


  else:
    # load test data
    test_data = utilities.CustomImageDataset('data.csv', parsed_args.d, utilities.test_transforms())
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
        data = output.squeeze()

        size = int(numpy.sqrt(data.size(0))) # construct number of rows and columns for subplot
        x = y = int(size) + 1 if size % 1 != 0 else int(size) # check if number of filters can be arranged in subplots
        plt.figure(figsize=(20, 17))
        for j, activation in enumerate(data):
          plt.subplot(x, y, j+1) # use shape of filter to define subplot
          plt.imshow(activation, cmap='viridis') 
          plt.axis('off')
          plt.savefig(F'{mod._get_name()}_Activation.png')
        plt.close()
        break

if __name__ == "__main__":
  main()
  