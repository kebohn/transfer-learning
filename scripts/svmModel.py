from baseModel import BaseModel
from sklearn import svm
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import torch
import PIL
import numpy


class SVMModel(BaseModel):
  def __init__(self, device):
    super().__init__(device)
    self.model = svm.SVC(kernel='linear', decision_function_shape='ovr', max_iter=1000)
    self.scaler = StandardScaler()
    
  def apply_transformations(self, image):
    new_image = resize(numpy.asarray(image), (224,224)) # resize image to 224x224
    return new_image.flatten()


  def extract(self, path):
    image = PIL.Image.open(path, 'r').convert('RGB') # open image skip transparency channel
    return self.apply_transformations(image)


  def fit(self, X_train, y_train):
    print("Fit SVM Model...")
    self.model.fit(X_train, y_train)
      
        
  def predict(self, X_test, *args, **kwargs):
    X_test_std = self.scaler.transform(X_test.reshape(1, -1)) # standardize test with same metrics as train data
    [y_test] = self.model.predict(X_test_std) # returns only one element
    return y_test
  
  
  def fit_scaler(self, data):
    return self.scaler.fit_transform(data)
  
  
  def get_scaler(self):
    return self.scaler
  
  
  def step_iter(self, X_train, y_train, step):
    vals, idx_start, counts = numpy.unique(y_train, return_counts=True, return_index=True) # counts number of same classes and their indices
    samples_per_cat = len(y_train) / len(vals)
    n = step
    while(n <= samples_per_cat): # increase steps till no more samples are left
      X_train_filtered = []
      y_train_filtered = []
      temp_n = numpy.repeat(n, len(vals))
      for idx, i in enumerate(idx_start): # go over the start indices of the categories
        if counts[idx] < n: # check if we have enough samples per class
          temp_n[idx] = counts[idx] # take the maximum
        X_train_filtered.extend(X_train[i:i+temp_n[idx]])
        y_train_filtered.extend(y_train[i:i+temp_n[idx]])
      n += step
      yield (X_train_filtered, y_train_filtered, n - step)