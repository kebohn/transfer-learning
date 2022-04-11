import os
import shutil

source_dir = '/local/scratch/bohn/datasets/aircraft-raw'
target_dir = '/local/scratch/bohn/datasets/aircraft'
splits = ['test', 'train', 'val']
datasets = ['families', 'manufacturers', 'variants']

# create target dirs
for dataset in datasets:
  dataset_dir = F'{target_dir}/{dataset}'
  os.makedirs(dataset_dir, exist_ok=True)
  # create all category dirs
  with open(F'{source_dir}/{dataset}/{dataset}.txt') as f:
    cats = f.read().splitlines()
    for split in splits:
      split_dir = F'{dataset_dir}/{split}'
      os.makedirs(split_dir, exist_ok=True)
      for cat in cats:
        idx = cat.find('/')
        if idx > -1:
          cat = cat[:idx] + cat[idx+1:] # remove / characters
        cat_dir = F'{split_dir}/{cat}'
        os.makedirs(cat_dir, exist_ok=True)
    
      # read split file
      with open(F'{source_dir}/{dataset}/images_{dataset}_{split}.txt') as f_split:
        lines = f_split.readlines()
        for line in lines:
          line_split = line.split(' ')
          file_name = line_split[0] # get filename
          cat = ' '.join(line_split[1:]).rstrip('\n') # concat list of string again and remove trailing newline char
          idx = cat.find('/')
          if idx > -1:
            cat = cat[:idx] + cat[idx+1:] # remove / character
          shutil.copy(F'{source_dir}/images/{file_name}.jpg', F'{split_dir}/{cat}')


# change size of test and validation - should be always 10 for each cat












