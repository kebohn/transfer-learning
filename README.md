# master-bohn

## Transfer Learning in Small Image Databases

### Baseline

Run baseline script and specify metric you want to use ['cosine', 'mean', 'kNN', 'svm']
Feature will be saved locally by adding `--extract` to the command and retrieved locally when the parameter is not defined
The path that specifies where the train and test dataset is located must also be defined when running the script. 

#### Example Cosine distance:

```
python3 baseline.py --d /local/scratch/bohn/datasets/context_virus_RAW/train/ --d_test /local/scratch/bohn/datasets/context_virus_RAW/test/ --cosine --extract
```

#### Extract Validation Images from Training Data

The script `createDatasetCsv.py` creates two csv files containing the filepath and the respective category. The files can then be used with the `ImageDataset` Class in order to load the defined datasets in a batch for the model. The amount of validation samples can be controlled by adding the argument `--v_size`.

### Transfer Learing Approaches

The following approaches are implemented in `adaptive.py`:

#### Fine-Tuning
  1. The pre-trained network is fine-tuned on the training set of the small dataset, and finally used as a predictor for the test dataset.
  2. Feature Extraction of the pre-trained network and used for a shallow learning algo (e.g. svm)
##### Adaptive Network
  1. The parameters in the trained network will be frozen and a adapter network will be trained with the small dataset.
  2. Feature Extraction of the adapter network and used for a shallow learning algo (e.g. svm) 

