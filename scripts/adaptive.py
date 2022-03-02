from adaptiveModel import AdaptiveModel
import torch
from torchvision import models, transforms
from imageDataset import CustomImageDataset
import matplotlib.pyplot as plt
import time
import numpy
import argparse
import os
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
  parser = argparse.ArgumentParser(description='Baseline script for transfer learning')
  parser.add_argument('--d', type=dir_path, help='Directory where files are stored (absolute dir)')
  parser.add_argument('--d_test', type=dir_path, help='Directory where test files are stored (absolute dir)')
  parser.add_argument('--model', type=dir_path, help='Directory where model is stored (absolute dir)')
  return parser.parse_args()


def dir_path(path):
  if os.path.isabs(path):
    return path
  raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")


def train_transforms():
  return transforms.Compose([
    transforms.Resize(224), # otherwise we would loose image information at the border
    transforms.CenterCrop(224), # take only center from image
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), # image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),  # scale pixel values to range [-3,3]

  ])

def test_transforms():
  return transforms.Compose([
    transforms.Resize(224), # otherwise we would loose image information at the border
    transforms.CenterCrop(224), # take only center from image
    transforms.ToTensor(), # image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),  # scale pixel values to range [-3,3]
  ])


def save_plot(x, y, x_label, y_label, title):
  plt.figure()
  for data in y.values():
    plt.plot(x, data)
  plt.xlabel(x_label) 
  plt.ylabel(y_label)
  plt.legend(list(y.keys()), loc='upper left')
  plt.savefig(F'{title}.jpg')


def define_model(data):
  # define network
  model = AdaptiveModel(data.get_categories()) # load pretrained model resnet-50

  # Freeze all layers except the additional classifier layers
  for name, param in model.named_parameters():
    if name.split('.')[0] not in 'classifier':
        param.requires_grad = False

  # add additional layer on top of pre-trained model which must be fine-tuned by small dataset
  #model.fc = torch.nn.Sequential(
  #  torch.nn.Linear(2048, 512), # output is number of classes in dataset
  #  torch.nn.Linear(512, data.get_categories()) # output is number of classes in dataset
  #  )

  return model.to(device) # save to GPU


def train(model, epochs, lr, momentum, train_loader, valid_loader, path):
  loss = torch.nn.CrossEntropyLoss()
  optimizer = optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
  valid_loss = []
  valid_acc = []
  train_loss = []
  train_acc = []
  num_correct = 0
  num_samples = 0

  # train network
  print("Learning Model...")
  since = time.time()
  for epoch in range(epochs):

    print(F'\nEpoch {epoch + 1}/{epochs}:')
    epoch_loss = 0

    for data, targets in train_loader: # iterate over training data in batches
      data = data.to(device)
      targets = targets.to(device)

      # forward pass
      scores = model(data)
      current_loss = loss(scores, targets)

      # backward pass
      optimizer.zero_grad()
      current_loss.backward()

      # gradient descent
      optimizer.step()

      # add current loss to epoch loss
      epoch_loss += current_loss.item()

      _, predictions = scores.max(1)
      num_correct += predictions.eq(targets).sum().item()
      num_samples += predictions.size(0)

    # compute training loss and accuracy and append to list
    train_loss.append(epoch_loss / len(train_loader))
    train_acc.append(num_correct / num_samples)
    print(F'Train Loss: {train_loss[-1]:.2f} | Accuracy: {train_acc[-1]:.2f}')


    # validate network
    print("Validate model...")
    epoch_loss = 0
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
      for x, y in valid_loader:
        x = x.to(device)
        y = y.to(device)

        # forward pass
        scores = model(x)

        _, predictions = scores.max(1)
        num_correct += predictions.eq(y).sum().item()
        num_samples += predictions.size(0)

        # add current loss to epoch loss
        current_loss = loss(scores, y)
        epoch_loss += current_loss.item()
        
    # compute test loss and accuracy and append to list
    valid_loss.append(epoch_loss / len(valid_loader))
    valid_acc.append(num_correct / num_samples)
    print(F'Validation Loss: {valid_loss[-1]:.2f} | Validation Accuracy: {valid_acc[-1]:.2f}')

  # save model
  path = path.rsplit('/', 2)
  torch.save(model.state_dict(), F'{path[0]}/model_lr_{lr}_epochs_{epochs}.pth')
    
  time_elapsed = time.time() - since
  print(F'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

  loss_data = {'train': train_loss, 'validation': valid_loss}
  acc_data = {'train': train_acc, 'validation': valid_acc}

  # write loss and accuracy to json
  with open('loss.json', 'w') as fp:
    json.dump(loss_data, fp,  indent=4)

  with open('acc.json', 'w') as fp:
    json.dump(acc_data, fp,  indent=4)

  # save loss
  save_plot(x=list(numpy.arange(1, epochs + 1)), y=loss_data, x_label='epochs', y_label='loss', title='Loss')
  # save accuracy
  save_plot(x=list(numpy.arange(1, epochs + 1)), y=acc_data, x_label='epochs', y_label='accuracy', title='Accuracy')


def test(model, test_loader):
  print("Test model...")
  num_correct = 0
  num_samples = 0
  model.eval()
  with torch.no_grad():
    for x, y in test_loader:
      x = x.to(device)
      y = y.to(device)

      # forward pass
      scores = model(x)

      _, predictions = scores.max(1)
      num_correct += predictions.eq(y).sum().item()
      num_samples += predictions.size(0)
        
  acc = num_correct / num_samples
  print(F'Test Accuracy: {acc:.2f}')
    
  # write accuracy to json
  with open('test_acc.json', 'w') as fp:
    json.dump(acc, fp,  indent=4)


def main():
  parsed_args = parse_arguments()

  # load data
  train_data = CustomImageDataset('data.csv', parsed_args.d, train_transforms())
  test_data = CustomImageDataset('data.csv', parsed_args.d_test, test_transforms())
 
  # Random split to generate validation data
  train_data_size = int(len(train_data) * 0.8)
  valid_data_size = len(train_data) - train_data_size
  train_data, valid_data = torch.utils.data.random_split(train_data, [train_data_size, valid_data_size], generator=torch.Generator().manual_seed(42))

  train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=8)
  valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=32, shuffle=True, num_workers=8)
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=8)

  # if specified saved model will be used otherwise a new model will be created
  if parsed_args.model is not None:
    model.load_state_dict(torch.load(parsed_args.model))
  else:
    model = define_model(test_data)

  epochs = 100
  lr = 0.01
  momentum = 0.9
  train(model, epochs, lr, momentum, train_loader, valid_loader, parsed_args.d)
  test(model, test_loader)

if __name__ == "__main__":
  main()