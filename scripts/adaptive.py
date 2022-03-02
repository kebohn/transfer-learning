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


def define_img_transforms():
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
  model = models.resnet50(pretrained=True) # load pretrained model resnet-50

  # Freeze model parameters
  for param in model.parameters():
    param.requires_grad = False

  # add additional layer on top of pre-trained model which must be fine-tuned by small dataset
  model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, data.get_categories()) # output is number of classes in dataset
    )

  return model.to(device) # save to GPU


def train(model, epochs, lr, momentum, train_loader, test_loader):
  loss = torch.nn.CrossEntropyLoss()
  optimizer = optimizer = torch.optim.SGD(params=model.fc.parameters(), lr=lr, momentum=momentum)
  test_loss = []
  test_acc = []
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
    print(F'Train Loss: {train_loss[-1]:.2f}| Accuracy: {train_acc[-1]:.2f}')


    # test network
    print("Testing Model with testing data...")
    epoch_loss = 0
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

        # add current loss to epoch loss
        current_loss = loss(scores, y)
        epoch_loss += current_loss.item()
        
    # compute test loss and accuracy and append to list
    test_loss.append(epoch_loss / len(test_loader))
    test_acc.append(num_correct / num_samples)
    print(F'Test Loss: {test_loss[-1]:.2f} | Test Accuracy: {test_acc[-1]:.2f}')

  # save model
  torch.save(model.state_dict(), F'/local/scratch/bohn/datasets/indoorCVPR_09/model_weights_lr_{lr}_epochs_{epochs}.pth')
    
  time_elapsed = time.time() - since
  print(F'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

  loss_data = {'train': train_loss, 'test': test_loss}
  acc_data = {'train': train_acc, 'test': test_acc}

  # write loss and accuracy to json
  with open('loss.json', 'w') as fp:
    json.dump(loss_data, fp,  indent=4)

  with open('acc.json', 'w') as fp:
    json.dump(acc_data, fp,  indent=4)

  # save loss
  save_plot(x=list(numpy.arange(1, epochs + 1)), y=loss_data, x_label='epochs', y_label='loss', title='Loss')
  # save accuracy
  save_plot(x=list(numpy.arange(1, epochs + 1)), y=acc_data, x_label='epochs', y_label='accuracy', title='Accuracy')


def main():
  parsed_args = parse_arguments()
  transform = define_img_transforms()

  # load data
  test_data = CustomImageDataset('data.csv', parsed_args.d_test, transform)
  test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True, num_workers=4)
  train_data = CustomImageDataset('data.csv', parsed_args.d, transform)
  train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)

  # if specified saved model will be used otherwise a new model will be created
  if parsed_args.model is not None:
    model.load_state_dict(torch.load(parsed_args.model))
  else:
    model = define_model(train_data)

  epochs = 25
  lr = 0.1
  momentum = 0.9
  train(model, epochs, lr, momentum, train_loader, test_loader)


if __name__ == "__main__":
  main()