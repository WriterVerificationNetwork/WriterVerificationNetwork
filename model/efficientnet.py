import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
import torch.nn.functional as F
from torch.nn import Identity
from torchvision.models import resnet, EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class EfficientNetB0(nn.Module):

    def __init__(self, numb_symbols, device, tasks, dropout=0.5):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        This network expects input size of 3 x 160 x 160
        For the Binary image reconstructing, it will have size 1 x 64 x 64
        """
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        backbone.classifier = Identity()
        self.backbone = backbone

        self.writer_footprint = nn.Sequential(
            nn.Linear(1280, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.symbol = nn.Sequential(
            nn.Linear(1280, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.symbol_classifier = nn.Sequential(
            nn.Linear(512, numb_symbols)
        )

        self.to(device)
        self._tasks = tasks

    def forward(self, x):
        x = self.backbone(x)
        results = {}

        if 'footprint' in self._tasks:
            footprint = self.writer_footprint(x)
            results['footprint'] = footprint

        return results


if __name__ == '__main__':
    from torchinfo import summary
    device = torch.device('cpu')
    model = EfficientNetB0(numb_symbols=24, device=device, tasks=['symbol', 'footprint'])
    batch_size = 10
    summary(model, input_size=(batch_size, 3, 224, 224), device=device)