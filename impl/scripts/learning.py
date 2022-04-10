#!/usr/bin/env python3
import sys
sys.path.append("..") # append the path of the parent directory

import torch
import torchvision
import copy
import utilities
import models
import data


def main():
  parsed_args = utilities.parse_arguments()
  current_size = parsed_args.step
  res = {}

  # load test and validation data
  print("Prepare test and validation dataset...")
  valid_data = data.CustomImageDataset('data.csv', parsed_args.d_valid, utilities.test_transforms())
  test_data = data.CustomImageDataset('data.csv', parsed_args.d_test, utilities.test_transforms())

  valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=10, shuffle=False, num_workers=8)
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=10 if parsed_args.adaptive else 1, shuffle=False, num_workers=8)

  # load different models
  print("Load models...")
  loaded_model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
  extraction_model = models.FEModel(model=loaded_model, device=utilities.get_device())
  adaptive_model = models.AdaptiveModel(num_categories=test_data.get_categories())

  # extract validation and test features
  valid_features = utilities.extract(extraction_model, valid_loader)
  test_features = utilities.extract(extraction_model, test_loader)

  # increase current size per category by step_size after every loop
  while(current_size <= parsed_args.max_size):
    print(F'Using {current_size} images per category...')

    # clear cache after each iteration
    torch.cuda.empty_cache()

    # load training data
    train_data = data.CustomImageDataset('data.csv', parsed_args.d, utilities.test_transforms() if parsed_args.pretrain else utilities.train_transforms(), current_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=False, num_workers=8)

    # training scheme only when we use the adaptive model or fine-tune the loaded model
    if parsed_args.adaptive or parsed_args.finetune:

      # define adapter model - must be always reinstantiated
      if parsed_args.adaptive:
        model = copy.deepcopy(adaptive_model) # must be always reinstantiated
    
      # prepare loaded model for fine-tuning
      else:
        model = copy.deepcopy(loaded_model) # must be always reinstantiated

        # replace last layer with the respective amount of dataset categories
        model.fc = torch.nn.Sequential(
          torch.nn.Linear(2048, test_data.get_categories()) # TODO replace number of neurons with variable when we use other loaded models
        )
    
        # set gradients to true in order to adapt the weights during training
        for param in model.parameters():
          param.requires_grad = True

      model.to(utilities.get_device()) # save to GPU

      # train data with current size of samples per category
      train_features_loader, _ = utilities.train(
        pre_trained_model=extraction_model,
        adapter_model=model, 
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_features=valid_features,
        params=parsed_args,
        current_size=current_size
      )

      # use Feature Extraction Model
      if parsed_args.adaptive:
      
        # normalize test data 
        test_features_norm = extraction_model.normalize_test(test_features)

        # handle test features like a dataset
        test_data = data.FeatureDataset(test_features_norm)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # extract features from model and use this with another specified metric to predict the categories
    if parsed_args.extract:

      if parsed_args.adaptive or parsed_args.finetune:

        # define trained adaptive extraction model
        learned_extraction_model = models.FEModel(model=model, device=utilities.get_device())

        # extract features from trained model
        tr_features = utilities.extract(learned_extraction_model, train_features_loader)
        te_features = utilities.extract(learned_extraction_model, test_loader)

      else:
        tr_features = utilities.extract(model, train_loader)
        te_features = utilities.extract(model, test_loader)
      
      # save features
      torch.save(tr_features, F'{parsed_args.results}features_train_size_{current_size}.pt')
      torch.save(te_features, F'{parsed_args.results}features_test_size_{current_size}.pt')

      # run prediction
      res[current_size] = utilities.predict(
          model=extraction_model if parsed_args.pretrain else learned_extraction_model,
          params=parsed_args,
          features=tr_features,
          test_loader=test_loader,
        )

    # use the model to classify the images
    else:
      res[current_size] = utilities.test(model, test_loader)

    current_size += parsed_args.step

  utilities.save_json_file(F'{parsed_args.results}res', res)
  utilities.save_training_size_plot(parsed_args.results, res)


if __name__ == "__main__":
  main()