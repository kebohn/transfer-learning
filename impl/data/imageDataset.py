import os
import PIL
import pandas as pd
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, samples=0):
        self.img_labels = pd.read_csv(F"{img_dir}{annotations_file}")
        self.current_size = samples
        self.size_has_changed = False
        # extract only defined number of samples per category from dataset
        if samples > 0:
            counts = self.img_labels['category'].value_counts() # count sampe sizes
            min_val = counts.min()
            if (min_val < samples): # check if every category has enough samples
                self.current_size = int(min_val)
                print(F"Fewer sample size in category found, current size changed to: {self.current_size}")
            self.img_labels = self.img_labels.groupby('category').head(self.current_size).reset_index(drop=True)
                
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = PIL.Image.open(img_path, 'r').convert(
            'RGB')  # open image skip transparency channel
        label = self.img_labels.iloc[idx, 1]
        name = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, label, name, self.img_labels.iloc[idx, 0]

    def get_categories(self):
        # count unique values in category column
        return self.img_labels['category'].nunique()

    def get_current_cat_size(self):
        return self.current_size, self.size_has_changed
