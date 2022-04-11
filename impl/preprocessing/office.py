import os
import random
import shutil

def copy_files(files, source, target):
  for file in files:
    shutil.copy(F'{source}/{file}', F'{target}/{file}')

source_dir = '/local/scratch/bohn/datasets/office_31-raw/amazon'
target_dir = '/local/scratch/bohn/datasets/office_31/amazon'
cats = os.listdir(source_dir)

# create train test and validation dirs
os.makedirs(F'{target_dir}/test', exist_ok=True)
os.makedirs(F'{target_dir}/train', exist_ok=True)
os.makedirs(F'{target_dir}/validation', exist_ok=True)

# create all category dirs
for new_dir in os.listdir(target_dir):
  for cat in cats:
    os.makedirs(F'{target_dir}/{new_dir}/{cat}', exist_ok=True)

for cat in cats:
  cat_dir = F'{source_dir}/{cat}'
  files = os.listdir(cat_dir)

  # retrieve 20 random samples for test and validation
  test_valid = random.sample(files, 20)

  # move first 10 files to new test directory
  copy_files(test_valid[:10], cat_dir, F'{target_dir}/test/{cat}')

  # move remaining files to new validation directory
  copy_files(test_valid[10:], cat_dir, F'{target_dir}/validation/{cat}')

  # get the training samples
  train_list = list(set(files) - set(test_valid))
  print(train_list)

  # move remaining files to new train directory
  copy_files(train_list, cat_dir, F'{target_dir}/train/{cat}')











