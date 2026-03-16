from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _weights(model_name: str, pretrained: bool):
    if not pretrained:
        return None

    weights_map = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "vgg19": models.VGG19_Weights.DEFAULT,
    }
    if model_name not in weights_map:
        raise ValueError(f"Unsupported backbone: {model_name}")
    return weights_map[model_name]


class ImageEncoder(nn.Module):
    FEATURE_DIMS = {
        "efficientnet_b0": 1280,
        "resnet50": 2048,
        "convnext_tiny": 768,
        "vgg19": 512,
    }

    def __init__(self, backbone_name: str, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = self._build_backbone(backbone_name, pretrained)
        self.feature_dim = self.FEATURE_DIMS[backbone_name]
        self.dropout = nn.Dropout(dropout)

    def _build_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        weights = _weights(backbone_name, pretrained)

        if backbone_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Identity()
            return model

        if backbone_name == "resnet50":
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
            return model

        if backbone_name == "convnext_tiny":
            model = models.convnext_tiny(weights=weights)
            model.classifier = nn.Identity()
            return model

        if backbone_name == "vgg19":
            model = models.vgg19(weights=weights)
            # Use a compact global pooling head instead of the huge default classifier.
            return nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.dropout(features)

    def _stage_modules(self) -> list[nn.Module]:
        if self.backbone_name == "efficientnet_b0":
            return list(self.backbone.features)

        if self.backbone_name == "resnet50":
            stem = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
            )
            return [stem, self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]

        if self.backbone_name == "convnext_tiny":
            return list(self.backbone.features)

        if self.backbone_name == "vgg19":
            features = self.backbone[0]
            return [features[:5], features[5:10], features[10:19], features[19:28], features[28:]]

        return [self.backbone]

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
