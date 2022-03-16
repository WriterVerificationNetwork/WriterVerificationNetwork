import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):

    def __init__(self, numb_symbols, device, tasks, dropout=0.5):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        This network expects input size of 3 x 160 x 160
        For the Binary image reconstructing, it will have size 1 x 64 x 64
        """
        super().__init__()
        self.encoding = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=2, padding=1, bias=False),  # 1 * 28 *28 to 8 * 14 * 14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, (3, 3), stride=2, padding=1, bias=False),  # 8 * 14 * 14 to 32 * 7 * 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, bias=False),  # 32 * 7 * 7 to 32 * 7 * 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)  # 32 * 7 * 7 to 32 * 7 * 7
        )

        self.decoding = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), padding=1, bias=False),  # 32 * 7 * 7 to 32 * 7 * 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), padding=1, bias=False),  # 32 * 7 * 7 to 32 * 7 * 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, (3, 3), stride=2, padding=1, output_padding=1, bias=False),  # 32 * 7 * 7 to 8 * 14 * 14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, (3, 3), stride=2, padding=1, output_padding=1)  # 8 * 14 * 14 to 1 * 28 * 28
        )

        self.footprint = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2)),    # 32 * 7 * 7 to 64 * 3 * 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64)
        )

        self.to(device)

        self._tasks = tasks

    def forward(self, x):
        x = self.encoding(x)

        results = {}

        if 'reconstruct' in self._tasks:
            # Binary image reconstructing
            reconstructing = self.decoding(x)
            results['reconstruct'] = reconstructing

        # Writer footprint declaration

        if 'footprint' in self._tasks:
            footprint = self.footprint(x)
            results['footprint'] = F.normalize(footprint, p=2)

        return results


if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cpu')
    model = SimpleNetwork(numb_symbols=24, device=device, tasks=['reconstruct', 'footprint'])
    batch_size = 10
    summary(model, input_size=(batch_size, 3, 28, 28), device=device)
