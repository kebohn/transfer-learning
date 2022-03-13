import shutil
import os

def move_files(directory, file_list, dir):
  with open(F"{directory}/{file_list}") as f:
    for line in f:
      file = line.strip()
      new_dir = file.split("/")[0]
      new_dir_path = F"{directory}/{dir}/{new_dir}"
      if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
      shutil.move(F"{directory}/Images/{file}", F"{directory}/{dir}/{file}")
      print(F"{file} listed in {file_list} moved to {new_dir_path}")

# move train and test files
directory = '/local/scratch/bohn/datasets/indoorCVPR_09'
move_files(directory, 'TrainImages.txt', 'train')
move_files(directory, 'TestImages.txt', 'test')
