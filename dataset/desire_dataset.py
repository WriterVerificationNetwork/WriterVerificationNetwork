import random
import torch
from torch.utils.data import Dataset


class DesireDataset(Dataset):

    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        return {
            'symbol': random.randint(0, 23),
            'img_anchor': torch.rand((3, 160, 160)),
            'bin_anchor': torch.rand((1, 64, 64)),
            'img_positive': torch.rand((3, 160, 160)),
            'bin_positive': torch.rand((1, 64, 64)),
            'img_negative': torch.rand((3, 160, 160)),
            'bin_negative': torch.rand((1, 64, 64))
        }
