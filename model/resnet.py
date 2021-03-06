import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNet18(nn.Module):

    def __init__(self, numb_symbols, device, tasks, dropout=0.5):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        This network expects input size of 3 x 160 x 160
        For the Binary image reconstructing, it will have size 1 x 64 x 64
        """
        super().__init__()

        backbone = resnet.__dict__['resnet18'](pretrained=True, norm_layer=FrozenBatchNorm2d)
        returned_layers = [1, 2]
        return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.pooling_layer = nn.MaxPool2d(2, 2)
        self.up_scaling_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.up_scaling_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up_scaling_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )

        self.final_reconstruct = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(1, 1))

        self.symbol_embedding = nn.Sequential(
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        self.symbol_embedding_projection = nn.Sequential(
            nn.Linear(512, 128)
        )

        self.final_symbol = nn.Linear(512, numb_symbols)

        self.writer_footprint = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        self.dropout = nn.Dropout(dropout)
        self.symbol_footprint_projection = nn.Linear(396, 128)
        self.to(device)

        self._tasks = tasks

    def forward(self, x):
        x = self.body(x)
        results = {}

        if 'reconstruct' in self._tasks:
            # Binary image reconstructing
            reconstructing = self.up_scaling_1(x['1'])
            reconstructing = self.up_scaling_2(reconstructing)
            reconstructing = self.up_scaling_3(reconstructing)
            reconstructing = self.final_reconstruct(reconstructing)
            results['reconstruct'] = reconstructing

        # Dimension reduction
        x = x['1']
        x = self.pooling_layer(x)
        x = x.view(x.shape[0], -1)

        # Symbol recognizing

        if 'symbol' in self._tasks:
            symbol_embedding = self.symbol_embedding(x)
            symbol = self.final_symbol(self.dropout(symbol_embedding))
            results['symbol'] = symbol

        # Writer footprint declaration

        if 'footprint' in self._tasks:
            footprint = self.writer_footprint(x)

            if 'symbol' in self._tasks:
                symbol_embedding_proj = self.symbol_embedding_projection(symbol_embedding)
                footprint = torch.cat([footprint, symbol_embedding_proj], dim=1)
                footprint = self.dropout(footprint)
                footprint = self.symbol_footprint_projection(footprint)
            results['footprint'] = footprint

        return results


if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cpu')
    model = ResNet18(numb_symbols=24, device=device, tasks=['reconstruct', 'symbol', 'footprint'])
    batch_size = 10
    summary(model, input_size=(batch_size, 3, 64, 64), device=device)
