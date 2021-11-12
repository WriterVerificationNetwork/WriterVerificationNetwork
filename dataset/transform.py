import torchvision.transforms


def get_transforms(args):
    return torchvision.transforms.Compose(
        torchvision.transforms.ToTensor()
    )


def reverse_transform():
    return torchvision.transforms.Compose(
        torchvision.transforms.ToPILImage()
    )