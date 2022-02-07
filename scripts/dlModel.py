from baseModel import BaseModel
import torch
import torchvision
import numpy
import collections, argparse, os, PIL


class DLModel(BaseModel):
  def __init__(self, device, params):
    super().__init__(device, params)
    self.model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
    modules = list(self.model.children())[:-1] # remove fully connected layer
    self.model = torch.nn.Sequential(*modules)
    self.model.eval() # evaluation mode
    self.model.to(device) # save on GPU

  def extract(self, path):
    with torch.no_grad(): # no training
        image = PIL.Image.open(path, 'r').convert('RGB') # open image skip transparency channel
        tensor = self.transform(image) # apply transformation defined above
        tensor = tensor.to(self.device) # save on GPU
        feature = self.model(tensor) # get model output
        return torch.flatten(feature).cpu()


  def predict(self, features):
    print("Scoring...")
    categories = collections.defaultdict(lambda: [0,0]) # store number of correct identifiactions and total number of identifications per category
    path = self.params.d

    if self.params.d_test is not None:
      path = self.params.d_test

    if self.params.neighbor:
      max_number_features = max([len(f) for f in features.values()]) # variable with maximum number of features for one category
      distances = numpy.zeros((len(features), max_number_features))
      labels = [] # all categories in training data for kNN algorithm

    for cat_dir in sorted(os.listdir(path)):
      cat = os.fsdecode(cat_dir)
      for file in sorted(os.listdir(F'{path}{cat}')):
        probe_feature = self.extract(F'{path}{cat}/{file}')
        min_distance = 1e8 # init high value
        index = 0
        for predicted_cat, feature in features.items():
      
          if self.params.cosine or self.params.neighbor:
            cosine_matrix = self.cos_similarity(feature.numpy(), probe_feature.numpy().reshape(1, -1)) # compute cosine of all existing features
            dist = 1.0 - numpy.max(cosine_matrix) # take the maximum similarity value and transform it to similarity distance
            if self.params.neighbor:
              distances[index, :len(feature)] = cosine_matrix.reshape(len(feature)) # persist all computed distances, will be used for kNN algo
              labels.append(predicted_cat) # persist all categories, will be used for kNN algo
          elif self.params.mean:
            dist = 1.0 - self.cos_similarity(torch.mean(feature, 0).numpy().reshape(1, -1), probe_feature.numpy().reshape(1, -1))[0,0] # compute similarity distance
          else:
            raise argparse.ArgumentTypeError('Metric not defined, use one of the following: (--mean, --cosine, --neighbor, --svm)')

          if dist < min_distance:
            min_distance = dist
            best_cat = predicted_cat

          index += 1

        if self.params.neighbor:
          k = self.params.k
          occurence_count = self.kNN(distances, k) # search the k-highest value

          if (len(numpy.where(occurence_count==occurence_count.max())) == 1 and k > 1): # check if a definitive winner has been found
            best_cat = labels[occurence_count.argmax()]
            
        categories[cat][0] += best_cat == cat # we only increase when category has been correctly identified
        categories[cat][1] += 1 # always increase after each iteration s.t. we have the total number
    
      # print rates for the current category
      print(F"Classified {categories[cat][0]:2} of {categories[cat][1]} images ({100 * categories[cat][0] / categories[cat][1]:6.2f}%) with {cat}")

    # print total accuracy
    total = numpy.sum(list(categories.values()), axis=0)
    print(F"\nAccuracy: {total[0]:3} of {total[1]} images ({100 * total[0] / total[1]:6.2f}%)")


  def fit(self):
    print("Fit function not yet implemented")