from abc import (
    ABC,
    abstractmethod
)
import torch
import torchvision


class BaseModel(ABC):
  def __init__(self, device):
    self.device = device
    self.transform = self.__define_img_transforms()


  def __define_img_transforms(self):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(224), # otherwise we would loose image information at the border
        torchvision.transforms.CenterCrop(224), # take only center from image
        torchvision.transforms.ToTensor(), # image to tensor
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),  # scale pixel values to range [-3,3]
        lambda x : x.unsqueeze(0) # required by pytorch (add batch dimension)
    ])
  
  @abstractmethod
  def extract(self):
    pass


  @abstractmethod
  def fit(self, X_train, y_train):
    pass


  @abstractmethod
  def predict(self, X_test):
    pass