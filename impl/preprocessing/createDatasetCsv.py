#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import os, argparse, csv
import torch
from sklearn import metrics
import numpy
import utilities
import data

def parse_arguments():
  parser = argparse.ArgumentParser(description='Creates a csv file from data')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--k', type=int, dest='k', default=-1, help='k-most similar images (computed with cosine distance) are used for the gallery (Default: -1)')
  return parser.parse_args()


def create_csv(path):
  '''Creates a ordered csv files with all images'''
  
  with open(F"{path}data.csv", "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["file", "category", "name"])
    y_label = 0 # transform string labels into numbers
    for cat_dir in sorted(os.listdir(path)):
      if not os.path.isdir(F'{path}{cat_dir}'):
        continue
      cat_name = os.fsdecode(cat_dir)
      for file_name in sorted(os.listdir(F'{path}{cat_name}')):
        file_dir = F"{cat_dir}/{file_name}"
        writer.writerow((file_dir, y_label, cat_name))
      y_label += 1


def create_k_best_gallery(img_loader, k, path): 
  '''Assembles the k most images per category by computing the similarity distance and saves it into a csv file.
  The remaining images are stored in a separate csv'''

  # files where result is stored
  with open(F"{path}gallery.csv", "w") as gallery_f, open(F"{path}data.csv", "w") as remaining_f:
    gallery_writer = csv.writer(gallery_f, delimiter=',')
    gallery_writer.writerow(["file", "category", "name"])
    remaining_writer = csv.writer(remaining_f, delimiter=',')
    remaining_writer.writerow(["file", "category", "name"])

    for data, targets, labels, paths in img_loader: # whole img data in one batch -> only one iteration

      # reshape images into 2D tensor
      all_samples = torch.reshape(data, (data.size(0),-1))

      # compute cosine similarity matrix
      sim = metrics.pairwise.cosine_similarity(all_samples)

      # ignore similarity values from same sample
      numpy.fill_diagonal(sim, 0.0)

      # get amount of classes
      cat_size = len(numpy.unique(labels))

      # counter variable
      current_pos = 0

      for i in range(cat_size):

        # get indices for current category
        indices = numpy.argwhere(targets.numpy() == i).flatten()
        sample_size = len(indices)

        # get only paths from current category 
        cat_paths = numpy.array(paths)[indices]

        # get only similarity score from current category
        sim_cat = sim[current_pos:current_pos + sample_size,current_pos:current_pos + sample_size]

        if k > 1:
          # get the k-1 indices with largest scores along each row aka image
          k_max = numpy.argpartition(-sim_cat, k-1, axis=1)[:,k-2::-1]

          # define row indices with k-1 repeats 
          row_idx = numpy.arange(sample_size)
          row_idx = numpy.repeat(row_idx, k-1)

          # get all respective scores
          scores = sim_cat[row_idx, k_max.ravel()].reshape(sample_size, k-1)

          # compute mean score per feature sample with the best k-1 scores
          mean_scores = numpy.mean(scores, axis=1)
        
          # get idx with max mean score - this is automatically the best sample 
          max_idx = numpy.argmax(mean_scores)

          # retrieve the indices which resulted in the best k-1 scores
          best_k_indices = k_max[max_idx, :]

          # append the best row index to list
          best_k_indices = numpy.append(best_k_indices, max_idx)
        else:
          # compute mean score per feature sample
          mean_scores = numpy.mean(sim_cat, axis=1)

          # get idx with max mean score - this is automatically the best sample 
          best_k_indices = numpy.argmax(mean_scores)

        all_idx = numpy.arange(sample_size)
        remaining_idx = numpy.delete(all_idx, best_k_indices)

        # add row for the chosen images
        for idx in best_k_indices:
          gallery_writer.writerow((cat_paths[idx], i, numpy.array(labels)[current_pos]))

        # add row for the not chosen images
        for idx in remaining_idx:
          remaining_writer.writerow((cat_paths[idx], i, numpy.array(labels)[current_pos]))
   
        current_pos += sample_size


def main():
  parsed_args = parse_arguments()
  if parsed_args.k > -1:
    # get data and init loader with whole data at once
    img_data = data.CustomImageDataset('data.csv', parsed_args.d, utilities.test_transforms())
    img_loader = torch.utils.data.DataLoader(dataset=img_data, batch_size=len(img_data), shuffle=False)
    create_k_best_gallery(img_loader, parsed_args.k, parsed_args.d)
  else:
    create_csv(parsed_args.d)


if __name__ == "__main__":
  main()
  