import os
import random
import shutil

def move_files(files, source, target):
  for file in files:
    shutil.move(F'{source}/{file}', F'{target}/{file}')

source_dir = '/local/scratch/bohn/datasets/office_31/amazon'
target_dir = '/local/scratch/bohn/datasets/office_31/amazon_cat'
cats = os.listdir(source_dir)

# create target dirs
try:
  # create train test and validation dirs
  os.mkdir(target_dir)
  os.mkdir(F'{target_dir}/test')
  os.mkdir(F'{target_dir}/train')
  os.mkdir(F'{target_dir}/validation')

  # create all category dirs
  for new_dir in os.listdir(target_dir):
    for cat in cats:
      os.mkdir(F'{target_dir}/{new_dir}/{cat}')
except:
  pass # swallow error


for cat in cats:
  cat_dir = F'{source_dir}/{cat}'
  files = os.listdir(cat_dir)

  # retrieve 20 random samples for test and validation
  test_valid = random.sample(files, 20)

  # move first 10 files to new test directory
  move_files(test_valid[:10], cat_dir, F'{target_dir}/test/{cat}')

  # move remaining files to new validation directory
  move_files(test_valid[10:], cat_dir, F'{target_dir}/validation/{cat}')

  # get the training samples
  files = os.listdir(cat_dir)

  # move remaining files to new train directory
  move_files(files, cat_dir, F'{target_dir}/train/{cat}')











