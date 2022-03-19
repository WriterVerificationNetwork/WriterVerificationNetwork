import numpy as np
import torchvision.transforms
import torchvision.transforms
from PIL import ImageOps
from PIL.Image import Image
from imgaug import augmenters as iaa


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
        ], p=applying_percent
        ),
        torchvision.transforms.ToTensor()
    ])


def val_transforms(args):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


def reverse_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])



