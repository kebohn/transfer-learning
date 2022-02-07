from abc import (
    ABC,
    abstractmethod
)
import torch
import torchvision
import PIL
import numpy


class BaseModel(ABC):
  def __init__(self, device, params):
    self.device = device
    self.transform = self.__define_img_transforms()
    self.params = params


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


  def kNN(self, distances, k):
    idx = numpy.argpartition(distances.ravel(), distances.size - k)[-k:] # search the k highest values
    max_idxs = numpy.column_stack(numpy.unravel_index(idx, distances.shape))
    occurence_count = numpy.bincount(max_idxs[:,0]) 
    return occurence_count


  def cos_similarity(self, A, B):
    num = numpy.dot(A, B.T)
    p1 = numpy.sqrt(numpy.sum(A**2, axis=1))[:,numpy.newaxis]
    p2 = numpy.sqrt(numpy.sum(B**2, axis=1))[numpy.newaxis,:]
    return num / (p1 * p2)


  @abstractmethod
  def extract(self):
    pass


  @abstractmethod
  def fit(self):
    pass


  @abstractmethod
  def predict(self):
    pass