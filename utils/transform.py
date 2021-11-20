import torchvision.transforms
from torchvision import transforms


def get_transforms(args):
    return torchvision.transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        torchvision.transforms.ToTensor()
    ])


def reverse_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])



