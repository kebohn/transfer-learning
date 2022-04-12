import torch
import torchvision
import matplotlib.pyplot as plt
import time
import numpy
import utilities
import models


def train(
    pre_trained_model,
    adapter_model,
    params,
    current_size,
    valid_features={},
    train_loader=[],
    valid_loader=[]
):
    # define loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=adapter_model.parameters(), lr=params.lr, momentum=params.momentum)

    # define running arrays
    valid_loss = []
    valid_acc = []
    valid_auc = []
    train_loss = []
    train_acc = []
    num_correct = 0
    num_samples = 0

    # early stopping: when the validation error in the last n epochs does not change we quit training
    es_counter = 0
    es_threshold = 30

    # train network
    print("Learning Model...")
    since = time.time()

    # retrieve train and valid feature loader
    if params.adaptive:
        train_loader, valid_loader = utilities.prepare_features_adaptive_training(
            pre_trained_model, train_loader, valid_features)

    for epoch in range(params.epochs):

        print(F'\nEpoch {epoch + 1}/{params.epochs}:')
        epoch_loss = 0

        for data, targets, _, _ in train_loader:  # iterate over training data in batches
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
        train_loss.append(epoch_loss / len(train_loader))
        train_acc.append(num_correct / num_samples)
        print(
            F'Train Loss: {train_loss[-1]:.2f} | Accuracy: {train_acc[-1]:.2f}')

        # validate network
        print("Validate model...")
        epoch_loss = 0
        num_correct = 0
        num_samples = 0
        adapter_model.eval()

        with torch.no_grad():
            for x, y, _, _ in valid_loader:
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
        current_valid_loss = epoch_loss / len(valid_loader)
        valid_loss.append(current_valid_loss)
        valid_acc.append(num_correct / num_samples)
        print(
            F'Validation Loss: {valid_loss[-1]:.2f} | Validation Accuracy: {valid_acc[-1]:.2f}')

        if params.early_stop:

            if params.auc:

                # define feature extraction model for validation set
                fe_model = models.FEModel(
                    adapter_model, utilities.get_device())

                # get all validation features
                valid_features = utilities.extract(fe_model, valid_loader)
                train_features = utilities.extract(fe_model, train_loader)

                # compute area under the curve for validation features
                valid_auc.append(
                    1.0 - utilities.calculate_auc(train_features, valid_features))

                es_counter = incease_early_stop_counter(valid_auc, es_counter)

            else:
                es_counter = incease_early_stop_counter(valid_loss, es_counter)

            print(
                F"Current early stopping threshold counter {es_counter}/{es_threshold}")

            if es_counter == es_threshold:
                print("Model starts to overfit, training stopped")
                break

    # save model
    torch.save(adapter_model.state_dict(
    ), F'{params.results}model_size_{current_size}_lr_{params.lr}_epochs_{epoch + 1}.pth')

    time_elapsed = time.time() - since
    print(
        F'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    loss_data = {'train': train_loss, 'validation': valid_loss}
    acc_data = {'train': train_acc, 'validation': valid_acc}

    # write loss and accuracy to json
    utilities.save_json_file(
        F'{params.results}loss_size_{current_size}', loss_data)
    utilities.save_json_file(
        F'{params.results}acc_size_{current_size}', acc_data)

    # save loss
    save_model_plot(x=list(numpy.arange(1, epoch + 2)), y=loss_data, x_label='epochs',
                    y_label='loss', title=F'{params.results}Loss_size_{current_size}')
    # save accuracy
    save_model_plot(x=list(numpy.arange(1, epoch + 2)), y=acc_data, x_label='epochs',
                    y_label='accuracy', title=F'{params.results}Accuracy_size_{current_size}')

    # return loader in case of extraction mode
    return train_loader, valid_loader


def test(model, test_loader):
    print("Test model...")
    res = {}
    y_test_arr = []
    perdictions_arr = []
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            y_test_arr.extend(y.detach().cpu().tolist())
            x = x.to(utilities.get_device())
            y = y.to(utilities.get_device())

            # forward pass
            scores = model(x)

            _, predictions = scores.max(1)
            perdictions_arr.extend(predictions.detach().cpu().tolist())
            num_correct += predictions.eq(y).sum().item()
            num_samples += predictions.size(0)

    acc = num_correct / num_samples
    print(F'Test Accuracy: {acc:.2f}')
    res["total_acc"] = acc
    res["labels"] = y_test_arr
    res["predictions"] = perdictions_arr

    return res


def incease_early_stop_counter(args_arr, es_counter):
    print(F"Current validation early stopping metric: {args_arr[-1]:.4f}")
    new_counter = es_counter
    if len(args_arr) >= 2:
        if args_arr[-2] - args_arr[-1] <= 1e-6:
            new_counter += 1
    return new_counter


def save_model_plot(x, y, x_label, y_label, title):
    plt.figure()
    for data in y.values():
        plt.plot(x, data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(list(y.keys()), loc='upper left')
    plt.savefig(F'{title}.jpg')
    plt.close()
