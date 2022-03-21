import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms
from PIL import ImageOps
from PIL.Image import Image
from imgaug import augmenters as iaa
from torch import nn
from torchvision.transforms import transforms

normal = torch.distributions.Normal(0, 0.15)


def addNoise(x, device='cpu'):
    """
    We will use this helper function to add noise to some data.
    x: the data we want to add noise to
    device: the CPU or GPU that the input is located on.
    """
    return x + normal.sample(sample_shape=torch.Size(x.shape)).to(device)


def get_transforms(args):
    applying_percent = 0.3
    sometimes = lambda aug: iaa.Sometimes(applying_percent, aug)
    return torchvision.transforms.Compose([
        lambda x: np.asarray(x),
        iaa.Sequential([
            sometimes(iaa.GaussianBlur(sigma=(0.0, 0.1))),
            sometimes(iaa.CoarseDropout(0.02, size_percent=0.5)),
            sometimes(iaa.LinearContrast((0.4, 1.6)))
        ]).augment_image,
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ], p=applying_percent),
        torchvision.transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #
        # torchvision.transforms.RandomApply([
        #     lambda x: addNoise(x, x.device)
        # ], p=0.6)
    ])


def val_transforms(args):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def reverse_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])



