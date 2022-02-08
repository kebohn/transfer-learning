from baseModel import BaseModel
from sklearn import svm
import torch
import PIL


class SVMModel(BaseModel):
  def __init__(self, device):
    super().__init__(device)
    self.model = svm.SVC(kernel='linear', decision_function_shape='ovr', max_iter=1000)


  def extract(self, path):
    image = PIL.Image.open(path, 'r').convert('RGB') # open image skip transparency channel
    tensor = self.transform(image) # apply transformation defined in baseModel
    return torch.flatten(tensor)


  def fit(self, X_train, y_train):
    print("Fit SVM Model...")
    self.model.fit(X_train, y_train)
      
        
  def predict(self, X_test, *args, **kwargs):
    [y_test] = self.model.predict(X_test.reshape(1, -1)) # returns only one element
    return y_test