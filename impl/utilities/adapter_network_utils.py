import torch
import torchvision
import matplotlib.pyplot as plt
import time
import numpy
import models
import utilities


def define_model(data, fine_tune):
  # define network
  base_model = torchvision.models.resnet50(pretrained=True) # load pretrained model resnet-50
  model = models.AdaptiveModel(model=base_model, num_categories=data.get_categories(), shallow=not fine_tune)

  # if fine_tune is true the whole model will be trained again
  if not fine_tune:
    # Freeze all layers except the additional classifier layers
    for name, param in model.named_parameters():
      # train the adapter network
      if name.split('.')[0] not in 'classifier':
          param.requires_grad = False

  return model.to(utilities.get_device()) # save to GPU


def train(model, epochs, lr, momentum, train_loader, valid_loader, path, early_stop, current_size):
  loss = torch.nn.CrossEntropyLoss()
  optimizer = optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
  valid_loss = []
  valid_acc = []
  train_loss = []
  train_acc = []
  num_correct = 0
  num_samples = 0

  # early stopping: when the validation error in the last 10 epochs does not change we quit training
  es_counter = 0 
  es_threshold = 10

  # train network
  print("Learning Model...")
  since = time.time()
  # learning rate decay 1/10 when a plateau is reached, currently not used
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, verbose=True, min_lr=1e-6)
  for epoch in range(epochs):

    print(F'\nEpoch {epoch + 1}/{epochs}:')
    epoch_loss = 0

    for data, targets, _ in train_loader: # iterate over training data in batches
      data = data.to(utilities.get_device())
      targets = targets.to(utilities.get_device())

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
      for x, y, _ in valid_loader:
        x = x.to(utilities.get_device())
        y = y.to(utilities.get_device())

        # forward pass
        scores = model(x)

        _, predictions = scores.max(1)
        num_correct += predictions.eq(y).sum().item()
        num_samples += predictions.size(0)

        # add current loss to epoch loss
        current_loss = loss(scores, y)
        epoch_loss += current_loss.item()
        
    # compute test loss and accuracy and append to list
    current_valid_loss = epoch_loss / len(valid_loader)
    valid_loss.append(current_valid_loss)
    valid_acc.append(num_correct / num_samples)
    print(F'Validation Loss: {valid_loss[-1]:.2f} | Validation Accuracy: {valid_acc[-1]:.2f}')
  
    # early stopping
    if len(valid_loss) >= 2 and early_stop:
      if abs(valid_loss[-2] - current_loss) <= 1e-2 or current_valid_loss > valid_loss[-2]:
        es_counter += 1

      if es_counter == es_threshold:
        print("Model starts to overfit, training stopped")
        break

    # rate decay when validation loss is not changing (currently not used)
    #scheduler.step(epoch_loss / len(valid_loader))

  # save model
  path = path.rsplit('/', 2)
  torch.save(model.state_dict(), F'{path[0]}/model_size_{current_size}_lr_{lr}_epochs_{epoch + 1}.pth')
    
  time_elapsed = time.time() - since
  print(F'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

  loss_data = {'train': train_loss, 'validation': valid_loss}
  acc_data = {'train': train_acc, 'validation': valid_acc}

  # write loss and accuracy to json
  utilities.save_json_file(F'loss_size_{current_size}', loss_data)
  utilities.save_json_file(F'acc_size_{current_size}', acc_data)

  # save loss
  save_model_plot(x=list(numpy.arange(1, epoch + 2)), y=loss_data, x_label='epochs', y_label='loss', title=F'Loss_size_{current_size}')
  # save accuracy
  save_model_plot(x=list(numpy.arange(1, epoch + 2)), y=acc_data, x_label='epochs', y_label='accuracy', title=F'Accuracy_size_{current_size}')


def test(model, test_loader, current_size):
  print("Test model...")
  num_correct = 0
  num_samples = 0
  model.eval()
  with torch.no_grad():
    for x, y, _ in test_loader:
      x = x.to(utilities.get_device())
      y = y.to(utilities.get_device())

      # forward pass
      scores = model(x)

      _, predictions = scores.max(1)
      num_correct += predictions.eq(y).sum().item()
      num_samples += predictions.size(0)
        
  acc = num_correct / num_samples
  print(F'Test Accuracy: {acc:.2f}')
    
  utilities.save_json_file(F'test_acc_size_{current_size}', acc)


def save_model_plot(x, y, x_label, y_label, title):
  plt.figure()
  for data in y.values():
    plt.plot(x, data)
  plt.xlabel(x_label) 
  plt.ylabel(y_label)
  plt.legend(list(y.keys()), loc='upper left')
  plt.savefig(F'{title}.jpg')
