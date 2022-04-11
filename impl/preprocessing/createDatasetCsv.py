#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import os, argparse, csv
import utilities

def parse_arguments():
  parser = argparse.ArgumentParser(description='Creates a csv file from data')
  parser.add_argument('--d', type=utilities.dir_path, help='Directory where files are stored (absolute dir)')
  return parser.parse_args()


def create_csv(path):
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


def main():
  parsed_args = parse_arguments()
  create_csv(parsed_args.d)


if __name__ == "__main__":
  main()
  