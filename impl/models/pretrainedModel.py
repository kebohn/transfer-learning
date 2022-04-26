import torchvision
import torch
import argparse


def get_pretrained_model(model_type):
    if model_type == 'resnet50':
        return torchvision.models.resnet50(pretrained=True)
    if model_type == 'alexnet':
        return torchvision.models.alexnet(pretrained=True)
    if model_type == 'vgg16':
        return torchvision.models.vgg16(pretrained=True)
    if model_type == 'mobilenet':
        return torchvision.models.mobilenet_v3_large(pretrained=True, width_mult=1.0,  reduced_tail=False, dilated=False)
    if model_type == 'densenet':
        return torchvision.models.densenet121(pretrained=True)
    raise argparse.ArgumentTypeError(
        f"model type: {model_type} is not a valid model")


def set_param_gradient(layer, val):
    for param in layer.parameters():
        param.requires_grad = val


def update_last_layer(model, model_type, category_size):
    if model_type == 'resnet50':
        model.fc = torch.nn.Linear(model.fc.in_features, category_size)
        #set_param_gradient(model.fc, True)
    else:
        if isinstance(model.classifier, torch.nn.Sequential):
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, category_size)
            #set_param_gradient(model.classifier[-1], True)
        else:
            model.classifier = torch.nn.Linear(model.classifier.in_features, category_size)
            #set_param_gradient(model.classifier, True)


def get_last_layer(model, model_type):
    if model_type == 'resnet50':
        return model.fc
    if isinstance(model.classifier, torch.nn.Sequential):
        return model.classifier[-1]
    return model.classifier
