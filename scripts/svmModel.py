from baseModel import BaseModel
from sklearn import svm
import torch
import torchvision
import PIL


class SVMModel(BaseModel):
    def __init__(self, device, params):
        super().__init__(device, params)
        self.model = svm.SVC(kernel='linear', decision_function_shape='ovr', max_iter=1000)
        self.model.to(device) # save on GPU


    def extract(self, path):
        with torch.no_grad(): # no training
            image = PIL.Image.open(path, 'r').convert('RGB') # open image skip transparency channel
            tensor = self.transform(image) # apply transformation defined above
            tensor = tensor.to(self.device) # save on GPU
            return torch.flatten(tensor).cpu()

    def predict(self):
        pass