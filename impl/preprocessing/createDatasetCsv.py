#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import os, argparse, csv
import utilities

def parse_arguments():
  parser = argparse.ArgumentParser(description='Creates a csv file from data')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--v_size', dest='v_size', type=int, default=0, help='Size of validation set per category (Default: 0)')
  return parser.parse_args()


def create_csv(path, v_size):
  with open(F"{path}data.csv", "w") as train_f, open(F"{path}validation.csv", "w") as valid_f:
    train_writer = csv.writer(train_f, delimiter=',')
    valid_writer = csv.writer(valid_f, delimiter=',')
    train_writer.writerow(["file", "category", "name"])
    valid_writer.writerow(["file", "category", "name"])
    y_label = 0 # transform string labels into numbers
    for cat_dir in sorted(os.listdir(path)):
      if not os.path.isdir(F'{path}{cat_dir}'):
        continue
      sample_count = 0
      cat_name = os.fsdecode(cat_dir)
      for file_name in sorted(os.listdir(F'{path}{cat_name}')):
        file_dir = F"{cat_dir}/{file_name}"
        if (sample_count < v_size):
          valid_writer.writerow((file_dir, y_label, cat_name))
        else:
          train_writer.writerow((file_dir, y_label, cat_name))
        sample_count += 1
      y_label += 1


def main():
  parsed_args = parse_arguments()
  create_csv(parsed_args.d, parsed_args.v_size)


if __name__ == "__main__":
  main()
  