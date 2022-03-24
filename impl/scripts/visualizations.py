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
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import utilities
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

def softmax(x):
  return numpy.exp(x) / sum(numpy.exp(x))


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
      features_test = torch.load(parsed_args.features_test)
      y_test = []
      for ctn, (_, val) in enumerate(features_test.items()):
        key_list = numpy.repeat(ctn, val.size(0)) # add the same amount of category label to list as samples
        y_test.extend(key_list) # add the current category label list to whole label list

      # one hot encoding of test labels
      y = label_binarize(y_test, classes=list(set(y_test)))
      
      # normalize all features
      features_test_norm = normalize_features(features_test)

      # combine features into one tensor
      features_test_norm_tensor = torch.cat(tuple(features_test_norm.values()), dim=0)

      # compute cosine similarity matrix
      sim = cosine_similarity(features_test_norm_tensor.cpu())

      # create probabilty score matrix for each sample
      sim_proba = numpy.zeros((y.shape[0], y.shape[1]))

      # ignore the similarity values on the diagonal because they will always be 1
      numpy.fill_diagonal(sim, 0.0)

      current_pos = 0

      # iterate over each class
      for idx, (cat, val) in enumerate(features_test_norm.items()):
        
        # get only similarity scores for one class
        sim_cat = sim[:, current_pos:(current_pos + val.size(0))]

        # get maximum value per feature sample (row)
        max_sim_cat = numpy.max(sim_cat, axis=1)

        print(max_sim_cat.shape)

        # add the computed maximum similarity scores to already initialized probalibity matrix
        sim_proba[:, idx] = max_sim_cat

        # increase current pos
        current_pos += val.size(0)

      # apply softmax on each row such that we have a probability for each class that sums up to 1
      sim_proba = numpy.apply_along_axis(softmax, 1, sim_proba)
      print(sim_proba.shape)
      print(y.shape)

      auc = roc_auc_score(y, sim_proba, multi_class='ovr', average=None)
      print(auc)

      #plt.figure()

      # iterate over all classes
      #fpr = tpr = thresh = roc_auc = dict()
      #for idx, (cat, vals) in enumerate(features.items()):

        # compute pairwise similarity of all features in one class
        #sim_self = cosine_similarity(vals.cpu())

        #max_sim_self = sim_self.max(axis=1)

        # filter out the current class
        #features_other = {k:v for k,v in features.items() if k != cat}

        # concat all other features in one tensor
        #vals_other = torch.cat(tuple(features_other.values()), dim=0) # combine features into one tensor

        # compute pairwise similarity with all other classes
        #sim_other = cosine_similarity(vals.cpu(), vals_other.cpu())

        #max_sim_other = sim_other.max(axis=1) # retrieve the largest distance per positive sample
        #sim_matrix = cosine_similarity(vals_test.cpu(), vals.cpu())

        #max_dist = sim_matrix.max(axis=1) # retrieve the largest distance per test sample

        #print(max_dist)

        # perfom roc scheme
        #fpr[idx], tpr[idx], thresh[idx] = roc_curve(y[:, idx], max_dist)
        #roc_auc[idx] = auc(fpr[idx], tpr[idx])
        # plotting    
        #plt.plot(fpr[idx], tpr[idx], label=F'Class {idx} vs Rest')
        #break

        #vals_magnitude = torch.linalg.vector_norm(vals, ord=2, dim=1)
        #vals_magnitude_other = torch.linalg.vector_norm(vals_other, ord=2, dim=1)
        #print(vals_magnitude.cpu().reshape(1,-1))
        #print(vals_magnitude_other.cpu().reshape(1,-1))

        #bins = numpy.linspace(min(vals_magnitude_other.cpu()), max(vals_magnitude_other.cpu()), 10)

        #plt.figure()
        #plt.hist(vals_magnitude.cpu().reshape(1,-1), bins, histtype='step', fill=False, label='class')
        #plt.hist(vals_magnitude_other.cpu().reshape(1,-1), bins, histtype='step', fill=False, label='other')
        #plt.savefig(F"hist{idx}.png")

        

      #plt.plot([0, 1], [0, 1], color="navy",  linestyle="--")
      #plt.xlim([0.0, 1.0])
      #plt.ylim([0.0, 1.05])
      #plt.title('Multiclass ROC curve')
      #plt.xlabel('False Positive Rate')
      #plt.ylabel('True Positive rate')
      #plt.legend(loc='best')
      #plt.savefig('Multiclass ROC', dpi=300)
      #plt.close()


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
  