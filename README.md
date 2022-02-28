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
