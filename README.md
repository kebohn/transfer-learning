# master-bohn

## Transfer Learning in Small Image Databases

This thesis examines different transfer learning approaches for image classification.

### Transfer Learning Approaches

The following approaches are implemented in `learning.py`:

#### Pre-Training
  1. Feature Extraction of the pre-trained network and used for a shallow learning algo or similarity metric ['cosine', 'mean', 'knn', 'svm']
#### Fine-Tuning
  1. The pre-trained network is fine-tuned on the training set of the small dataset, and finally used as a predictor for the test dataset.
  2. Feature Extraction of the pre-trained network and used for a shallow learning algo or similarity metric ['cosine', 'mean', 'knn', 'svm']
#### Adaptive Network
  1. The parameters in the trained network will be frozen and a adapter network will be trained with the small dataset.
  2. Feature Extraction of the adapter network and used for a shallow learning algo or similarity metric ['cosine', 'mean', 'knn', 'svm']

### Deep Features Gallery Approaches
Different approaches when using the deep features in the image classification task are available:
1. Use gradually more training samples per category for training and also use all training data to extract the features for classification → all training samples are then features for the gallery. This is the default option. No further parameters must be added.
2. Use gradually more training samples per category for training the network but use always the same 5 samples per category in the gallery for classification that are not part of the training. In this case the script `createDatasetCsv.py --d path-to-training-data --k 5` must be executed. This creates two separate csv files: `data.csv` will then be used for training the model and `gallery.csv` will be used as a permanent gallery. After creating the csv files the main script `learning.py` with the additional parameter `--k-gallery` must be set in order to use the permanent gallery images.
3. Use all training samples per category for training and then gradually increase the gallery samples.


#### Example Pre-Trained Approach with extracted Deep Features using Cosine distance:

```
python3 learning.py --d /local/scratch/bohn/datasets/context_virus/train/ --d-test /local/scratch/bohn/datasets/context_virus/test/ --d-validation /local/scratch/bohn/datasets/context_virus/test/ --pretrain --cosine --extract
```

Using other approaches and metrics explore the functionality by hitting ```python3 learning.py --help```

#### Preprocessing

The script `createDatasetCsv.py` creates a csv file containing the filepath and the respective category. The files can then be used with the `ImageDataset` Class in order to load the defined datasets in a batch for the model.

#### Datasets

The following Datasets are used to examine the approaches:
- Aircraft https://arxiv.org/pdf/1306.5151.pdf

- Virus https://www.sciencedirect.com/science/article/pii/S0169260721003928

- Fruit & Vegetables https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition

- Indoor http://wb.mit.edu/torralba/www/indoor.html

- Office https://link.springer.com/chapter/10.1007/978-3-642-15561-1_16
