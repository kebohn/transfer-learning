import torch
import torchvision
import matplotlib.pyplot as plt
import time
import numpy
import utilities



def train(
  pre_trained_model,
  adapter_model,
  epochs,
  lr,
  momentum,
  parsed_args,
  current_size,
  features_valid={},
  train_loader=[],
  valid_loader=[]
):
  # define loss and optimizer
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=adapter_model.parameters(), lr=lr, momentum=momentum)

  # define running arrays
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
 
  for epoch in range(epochs):

    print(F'\nEpoch {epoch + 1}/{epochs}:')
    epoch_loss = 0

    # retrieve train and valid feature loader
    if features_valid:
      t_loader, v_loader, f_train = utilities.prepare_features_for_training(pre_trained_model, train_loader, features_valid)
      
      # save features
      torch.save(f_train, F'{parsed_args.results}features_pretrained_size_{current_size}.pt')
    else:
      t_loader, v_loader = train_loader, valid_loader


    for data, targets, _ in t_loader: # iterate over training data in batches
      data = data.to(utilities.get_device())
      targets = targets.to(utilities.get_device())

      optimizer.zero_grad()

      # forward pass
      scores = adapter_model(data)
      current_loss = loss(scores, targets)

      # backward pass
      current_loss.backward()

      # gradient descent
      optimizer.step()

      # add current loss to epoch loss
      epoch_loss += current_loss.item()

      _, predictions = scores.max(1)
      num_correct += predictions.eq(targets).sum().item()
      num_samples += predictions.size(0)

    # compute training loss and accuracy and append to list
    train_loss.append(epoch_loss / len(t_loader))
    train_acc.append(num_correct / num_samples)
    print(F'Train Loss: {train_loss[-1]:.2f} | Accuracy: {train_acc[-1]:.2f}')

    # validate network
    print("Validate model...")
    epoch_loss = 0
    num_correct = 0
    num_samples = 0
    adapter_model.eval()

    with torch.no_grad():
      for x, y, _ in v_loader:
        x = x.to(utilities.get_device())
        y = y.to(utilities.get_device())

        # forward pass
        scores = adapter_model(x)

        _, predictions = scores.max(1)
        num_correct += predictions.eq(y).sum().item()
        num_samples += predictions.size(0)

        # add current loss to epoch loss
        current_loss = loss(scores, y)
        epoch_loss += current_loss.item()
        
    # compute test loss and accuracy and append to list
    current_valid_loss = epoch_loss / len(v_loader)
    valid_loss.append(current_valid_loss)
    valid_acc.append(num_correct / num_samples)
    print(F'Validation Loss: {valid_loss[-1]:.2f} | Validation Accuracy: {valid_acc[-1]:.2f}')
  
    # early stopping
    if len(valid_loss) >= 2 and parsed_args.early_stop:
      if abs(valid_loss[-2] - current_loss) <= 1e-2 or current_valid_loss > valid_loss[-2]:
        es_counter += 1

      if es_counter == es_threshold:
        print("Model starts to overfit, training stopped")
        break

  # save model
  torch.save(adapter_model.state_dict(), F'{parsed_args.results}model_size_{current_size}_lr_{lr}_epochs_{epoch + 1}.pth')
    
  time_elapsed = time.time() - since
  print(F'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

  loss_data = {'train': train_loss, 'validation': valid_loss}
  acc_data = {'train': train_acc, 'validation': valid_acc}

  # write loss and accuracy to json
  utilities.save_json_file(F'{parsed_args.results}loss_size_{current_size}', loss_data)
  utilities.save_json_file(F'{parsed_args.results}acc_size_{current_size}', acc_data)

  # save loss
  save_model_plot(x=list(numpy.arange(1, epoch + 2)), y=loss_data, x_label='epochs', y_label='loss', title=F'{parsed_args.results}Loss_size_{current_size}')
  # save accuracy
  save_model_plot(x=list(numpy.arange(1, epoch + 2)), y=acc_data, x_label='epochs', y_label='accuracy', title=F'{parsed_args.results}Accuracy_size_{current_size}')

  # return loader in case of extraction mode
  return t_loader, v_loader


def test(model, test_loader):
  print("Test model...")
  res = {}
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
  res["total_acc"] = acc
    
  return res


def save_model_plot(x, y, x_label, y_label, title):
  plt.figure()
  for data in y.values():
    plt.plot(x, data)
  plt.xlabel(x_label) 
  plt.ylabel(y_label)
  plt.legend(list(y.keys()), loc='upper left')
  plt.savefig(F'{title}.jpg')
  plt.close()
