import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
import torch.nn.functional as F


class WriterVerificationNetwork(InceptionResnetV1):

    def __init__(self, numb_symbols, device, tasks, dropout=0.5):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        This network expects input size of 3 x 160 x 160
        For the Binary image reconstructing, it will have size 1 x 64 x 64
        """
        super().__init__(classify=False, device=device, num_classes=1, dropout_prob=dropout)

        self.up_scaling_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=896, out_channels=256, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.up_scaling_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_scaling_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )

        self.final_reconstruct = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1))

        self.symbol_embedding = nn.Sequential(
            nn.Linear(1792, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.symbol_embedding_projection = nn.Sequential(
            nn.Linear(256, 128)
        )

        self.final_symbol = nn.Linear(256, numb_symbols)

        self.writer_footprint = nn.Sequential(
            nn.Linear(1792, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256)
        )
        self.symbol_footprint_projection = nn.Linear(640, 256)
        self.to(device)

        self._tasks = tasks

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)

        results = {}

        if 'reconstruct' in self._tasks:
            # Binary image reconstructing
            reconstructing = self.up_scaling_1(x)
            reconstructing = self.up_scaling_2(reconstructing)
            reconstructing = self.up_scaling_3(reconstructing)
            reconstructing = self.final_reconstruct(reconstructing)
            results['reconstruct'] = reconstructing

        # Dimension reduction
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        # x = self.dropout(x)
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
            results['footprint'] = F.normalize(footprint, p=2)

        return results


if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cpu')
    model = WriterVerificationNetwork(numb_symbols=24, device=device, tasks=['reconstruct', 'symbol', 'footprint'])
    batch_size = 10
    summary(model, input_size=(batch_size, 3, 160, 160), device=device)
