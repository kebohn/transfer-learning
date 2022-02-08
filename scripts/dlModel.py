from baseModel import BaseModel
import torch
import torchvision
import numpy
import collections, argparse, PIL


class DLModel(BaseModel):
  def __init__(self, device):
    super().__init__(device)
    self.model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
    modules = list(self.model.children())[:-1] # remove fully connected layer
    self.model = torch.nn.Sequential(*modules)
    self.model.eval() # evaluation mode
    self.model.to(device) # save on GPU
    
    
  def extract(self, path):
    with torch.no_grad(): # no training
        image = PIL.Image.open(path, 'r').convert('RGB') # open image skip transparency channel
        tensor = self.transform(image) # apply transformation defined in baseModel
        tensor = tensor.to(self.device) # save on GPU
        feature = self.model(tensor) # get model output
        return torch.flatten(feature).cpu()


  def predict(self, X_test, features, distances, labels, params):
    min_distance = 1e8 # init high value
    index = 0
    for predicted_cat, feature in features.items():
      if params.cosine or params.neighbor:
        cosine_matrix = self.__cos_similarity(feature.numpy(), X_test.numpy().reshape(1, -1)) # compute cosine of all existing features
        dist = 1.0 - numpy.max(cosine_matrix) # take the maximum similarity value and transform it to similarity distance
        if params.neighbor:
          distances[index, :len(feature)] = cosine_matrix.reshape(len(feature)) # persist all computed distances, will be used for kNN algo
          labels.append(predicted_cat) # persist all categories, will be used for kNN algo
      elif params.mean:
        dist = 1.0 - self.__cos_similarity(torch.mean(feature, 0).numpy().reshape(1, -1), X_test.numpy().reshape(1, -1))[0,0] # compute similarity distance
      else:
        raise argparse.ArgumentTypeError('Metric not defined, use one of the following: (--mean, --cosine, --neighbor --svm)')

      if dist < min_distance:
        min_distance = dist
        best_cat = predicted_cat

      index += 1

    if params.neighbor:
      occurence_count = self.__kNN(distances, params.k) # search the k-highest value

      if (len(numpy.where(occurence_count==occurence_count.max())) == 1 and params.k > 1): # check if a definitive winner has been found
        best_cat = labels[occurence_count.argmax()]
        
    return best_cat

  
  def fit(self):
    pass
  
  
  def __kNN(self, distances, k):
    idx = numpy.argpartition(distances.ravel(), distances.size - k)[-k:] # search the k highest values
    max_idxs = numpy.column_stack(numpy.unravel_index(idx, distances.shape))
    occurence_count = numpy.bincount(max_idxs[:,0]) 
    return occurence_count


  def __cos_similarity(self, A, B):
    num = numpy.dot(A, B.T)
    p1 = numpy.sqrt(numpy.sum(A**2, axis=1))[:,numpy.newaxis]
    p2 = numpy.sqrt(numpy.sum(B**2, axis=1))[numpy.newaxis,:]
    return num / (p1 * p2)