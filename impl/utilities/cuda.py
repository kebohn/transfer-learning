
import torch

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


def get_device():
    return device
