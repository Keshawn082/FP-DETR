"""Backbone: Programmable ResNet (PResNet) for FP-DETR.

Provides a lightweight backbone that returns multi-scale feature maps at
strides 8, 16, and 32 from a standard ResNet-18/50 (torchvision) or a
custom depth configuration.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class PResNet(nn.Module):
    """Programmable ResNet backbone for FP-DETR.

    Returns feature maps from layers 2, 3, and 4 (strides 8, 16, 32).

    Args:
        depth:    Backbone depth: 18 or 50.
        pretrained: Load ImageNet pretrained weights.
        freeze_at: Freeze the first N stages (0 = no freeze).
        return_idx: Indices of ResNet stages to return (0-indexed from layer1).
    """

    _out_channels = {
        18: [128, 256, 512],
        50: [512, 1024, 2048],
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        freeze_at: int = 0,
        return_idx: tuple = (1, 2, 3),
    ):
        super().__init__()
        assert depth in (18, 50), f"Unsupported depth {depth}. Choose 18 or 50."
        self.return_idx = return_idx

        if depth == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
        else:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet50(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self._out_ch = self._out_channels[depth]

        if freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad_(False)
        if freeze_at >= 2:
            for p in self.layer1.parameters():
                p.requires_grad_(False)

    @property
    def out_channels(self) -> list:
        return [self._out_ch[i - 1] for i in self.return_idx]

    def forward(self, x: torch.Tensor) -> list:
        out = []
        x = self.stem(x)
        x = self.layer1(x)
        if 0 in self.return_idx:
            out.append(x)
        x = self.layer2(x)
        if 1 in self.return_idx:
            out.append(x)
        x = self.layer3(x)
        if 2 in self.return_idx:
            out.append(x)
        x = self.layer4(x)
        if 3 in self.return_idx:
            out.append(x)
        return out
