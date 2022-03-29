import os
import argparse
import json


def dir_path(path):
  if os.path.isabs(path):
    return path
  raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")


def file_iterable(path):
  for cat_dir in sorted(os.listdir(path)):
    cat_name = os.fsdecode(cat_dir)
    if os.path.isfile(F'{path}{cat_name}'): # only consider directories
      continue
    for file_name in sorted(os.listdir(F'{path}{cat_name}')):
      yield (cat_name, file_name)


def save_json_file(name, content):
  with open(F"{name}.json", 'w') as fp:
        json.dump(content, fp,  indent=4)


def load_json_file(path):
  f = open(path, "r")
  data = json.loads(f.read())
  f.close()
  return data
