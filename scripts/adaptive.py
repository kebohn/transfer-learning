import torch
from torchvision import models, transforms
from imageDataset import CustomImageDataset
import matplotlib.pyplot as plt
import time
import numpy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


transform = define_img_transforms()

# load data
train_data = CustomImageDataset('data.csv', '/local/scratch/bohn/datasets/indoorCVPR_09/train/', transform)
test_data = CustomImageDataset('data.csv', '/local/scratch/bohn/datasets/indoorCVPR_09/test/', transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

# define network
model = models.resnet50(pretrained=True) # load pretrained model resnet-50

# Freeze model parameters
for param in model.parameters():
  param.requires_grad = False

# add additional layer on top of pre-trained model which must be fine-tuned by small dataset
model.fc = torch.nn.Sequential(
  torch.nn.Linear(2048, train_data.get_categories()) # output is number of classes in dataset
  )

model.to(device) # save to GPU

epochs = 25
lr = 0.1
momentum = 0.9

loss = torch.nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.SGD(params=model.fc.parameters(), lr=lr, momentum=momentum)
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

time_elapsed = time.time() - since
print(F'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


# test network
print("Testing Model with testing data...")
test_loss = []
test_acc = []
num_correct = 0
num_samples = 0
model.eval()

with torch.no_grad():
  for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)

    scores = model(x)
    _, predictions = scores.max(1)
    num_correct += predictions.eq(targets).sum().item()
    num_samples += predictions.size(0)

    
# compute test loss and accuracy and append to list
test_loss.append(epoch_loss / len(test_loader))
test_acc.append(num_correct / num_samples)
print(F'Train Loss: {test_loss[-1]:.2f} | Accuracy: {test_acc[-1]:.2f}')

# save loss
save_plot(x=list(numpy.arrange(epochs)), y={'train': train_loss, 'test': test_loss}, x_label='epochs', y_label='loss', title='Loss')
# save accuracy
save_plot(x=list(numpy.arrange(epochs)), y={'train': train_acc, 'test': test_acc}, x_label='epochs', y_label='accuracy', title='Accuracy')

# save model
torch.save(model.cpu().state_dict(), F'/local/scratch/bohn/datasets/indoorCVPR_09/model_weights_lr_{lr}_epochs_{epochs}.pth')