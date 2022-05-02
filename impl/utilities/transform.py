import torchvision


def train_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
        torchvision.transforms.ColorJitter(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),  # image to tensor
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),  # scale pixel values to range [-3,3]
    ])


def test_transforms():
    return torchvision.transforms.Compose([
        # otherwise we would loose image information at the border
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),  # take only center from image
        torchvision.transforms.ToTensor(),  # image to tensor
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),  # scale pixel values to range [-3,3]
    ])
