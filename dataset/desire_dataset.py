import random
import torch
from torch.utils.data import Dataset

from dataset import utils


class DesireDataset(Dataset):

    def __init__(self, gt_dir, gt_binarized_dir, transforms) -> None:
        super().__init__()
        self.gt_dir = gt_dir
        self.gt_binarized_dir = gt_binarized_dir
        self.transforms = transforms

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        # Expecting all images to have their dimension of 160 x 160 x 3
        # Expecting all binary images to have their dimension of 64 x 64 x 1
        return {
            'symbol': random.randint(0, len(utils.letters) - 1),
            'img_anchor': torch.rand((3, 160, 160)),
            'bin_anchor': torch.rand((1, 64, 64)),
            'img_positive': torch.rand((3, 160, 160)),
            'bin_positive': torch.rand((1, 64, 64)),
            'img_negative': torch.rand((3, 160, 160)),
            'bin_negative': torch.rand((1, 64, 64))
        }
