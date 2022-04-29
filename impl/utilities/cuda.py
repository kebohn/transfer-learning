
import torch

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


def get_device():
    return device
 