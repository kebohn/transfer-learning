import shutil
import os
import random


def move_files(directory, file_list, dir):
    with open(F"{directory}/{file_list}") as f:
        for line in f:
            file = line.strip()
            new_dir = file.split("/")[0]
            new_dir_path = F"{directory}/{dir}/{new_dir}"
            os.makedirs(new_dir_path, exist_ok=True)
            shutil.move(F"{directory}/Images/{file}",
                        F"{directory}/{dir}/{file}")
            print(F"{file} listed in {file_list} moved to {new_dir_path}")


# move train and test files
directory = '/local/scratch/bohn/datasets/indoorCVPR_09'
move_files(directory, 'TrainImages.txt', 'train')
move_files(directory, 'TestImages.txt', 'test')

# split train into validation samples

train_cats = os.listdir(F'{directory}/train')

for cat in train_cats:
    # create category dir
    os.makdirs(F'{directory}/validation/{cat}', exist_ok=True)

    files = os.listdir(F'{directory}/train/{cat}')

    # retrieve 10 random samples for validation
    test_valid = random.sample(files, 10)

    # move files to new validation directory
    for file in test_valid:
        shutil.move(F'{directory}/train/{cat}/{file}',
                    F'{directory}/validation/{cat}/{file}')
