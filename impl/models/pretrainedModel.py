import torchvision
import torch


def get_pretrained_model(model_type):
    if model_type == 'resnet50':
        return torchvision.models.resnet50(pretrained=True)
    if model_type == 'alexnet':
        return torchvision.models.alexnet(pretrained=True)
    if model_type == 'vgg16':
        return torchvision.models.vgg16(pretrained=True)
    if model_type == 'vgg19':
        return torchvision.models.vgg19(pretrained=True)
    if model_type == 'densenet':
        return torchvision.models.densenet121(pretrained=True)


def update_last_layer(model, model_type, category_size):
    if model_type == 'resnet50':
        model.fc = torch.nn.Linear(model.fc.in_features, category_size)
    else:
        if isinstance(model.classifier, torch.nn.Sequential):
            print(model.classifier[-1])
            model.classifier[-1] = torch.nn.Linear(
                model.classifier[-1].in_features, category_size)
        else:
            model.classifier = torch.nn.Linear(
                model.classifier.in_features, category_size)
