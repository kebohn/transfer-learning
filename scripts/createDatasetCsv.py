#!/usr/bin/env python3

import os, argparse, csv

def parse_arguments():
  parser = argparse.ArgumentParser(description='Creates a csv file from data')
  parser.add_argument('--d', type=dir_path, help='Directory where files are stored (absolute dir)')
  return parser.parse_args()


def dir_path(path):
  if os.path.isabs(path):
    return path
  raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")

def create_csv(path):
  with open(F"{path}data.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    y_label = 0 # transform string labels into numbers
    for cat_dir in sorted(os.listdir(path)):
      if not os.path.isdir(F'{path}{cat_dir}'):
        continue
      cat_name = os.fsdecode(cat_dir)
      for file_name in sorted(os.listdir(F'{path}{cat_name}')):
        writer.writerow((F"{cat_dir}/{file_name}", y_label)) 
      y_label += 1


def main():
  parsed_args = parse_arguments()
  create_csv(parsed_args.d)


if __name__ == "__main__":
  main()
  