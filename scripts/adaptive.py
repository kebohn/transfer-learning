import torch
from torchvision import models, transforms
from imageDataset import CustomImageDataset

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
  torch.nn.Linear(2048, 256),
  torch.nn.ReLU(inplace=True),
  torch.nn.Linear(256, len(train_loader)) # output is number of classes in dataset
  )

model.to(device) # save to GPU

epochs = 100
lr = 0.01
momentum = 0.9

loss = torch.nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

# train network
print("Learning Model...")
for epoch in range(epochs):
  for batch_idx, (data, targets) in enumerate(train_loader):
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


def check_accuracy(loader, model):
  num_correct = 0
  num_samples = 0
  model.eval()

  with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(F'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


# test network
print("Testing Model with training data...")
check_accuracy(train_loader, model)
print("Testing Model with testing data...")
check_accuracy(test_loader, model)