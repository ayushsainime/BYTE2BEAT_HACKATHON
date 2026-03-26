from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    FEATURE_DIMS = {
        "efficientnet_b4": 1792,
        "resnet50": 2048,
    }

    def __init__(self, backbone_name: str, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = self._build_backbone(backbone_name, pretrained)
        self.feature_dim = self.FEATURE_DIMS[backbone_name]
        self.dropout = nn.Dropout(dropout)

    def _build_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        if backbone_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            model.classifier = nn.Identity()
            return model

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
            return model

        raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.dropout(features)

    def _stage_modules(self) -> list[nn.Module]:
        if self.backbone_name == "efficientnet_b4":
            return list(self.backbone.features)

        stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        return [stem, self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]

    def freeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def unfreeze_last_n_stages(self, n: int) -> None:
        stages = self._stage_modules()
        self.freeze_all()
        for stage in stages[-n:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True
