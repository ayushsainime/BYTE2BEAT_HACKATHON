from __future__ import annotations

import torch
from torch import nn

from models.image_encoder import ImageEncoder
from utils.config import FreezePolicyConfig, ModelConfig


class MultimodalRiskModel(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

        self.image_encoder = ImageEncoder(
            backbone_name=model_config.backbone_name,
            pretrained=model_config.pretrained,
            dropout=model_config.image_feature_dropout,
        )

        self.metadata_branch = nn.Sequential(
            nn.Linear(1, model_config.metadata_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(model_config.metadata_dropout),
            nn.Linear(model_config.metadata_hidden_dim, model_config.metadata_hidden_dim),
            nn.ReLU(inplace=True),
        )

        fusion_input_dim = (self.image_encoder.feature_dim * 2) + model_config.metadata_hidden_dim
        self.fusion_head = self._build_fusion_head(
            input_dim=fusion_input_dim,
            hidden_dims=model_config.fusion_hidden_dims,
            num_labels=model_config.num_labels,
            dropout=model_config.fusion_dropout,
        )

    @staticmethod
    def _build_fusion_head(input_dim: int, hidden_dims: list[int], num_labels: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        current = input_dim

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden

        layers.append(nn.Linear(current, num_labels))
        return nn.Sequential(*layers)

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        left_features = self.image_encoder(left_image)
        right_features = self.image_encoder(right_image)
        metadata_features = self.metadata_branch(metadata)

        fused_features = torch.cat([left_features, right_features, metadata_features], dim=1)
        return self.fusion_head(fused_features)

    def apply_freeze_policy(self, epoch: int, freeze_policy: FreezePolicyConfig) -> None:
        if not freeze_policy.enabled:
            self.image_encoder.unfreeze_all()
            return

        if epoch < freeze_policy.freeze_encoder_epochs:
            self.image_encoder.freeze_all()
            return

        # After freeze epochs, fully unfreeze for fine-tuning.
        self.image_encoder.unfreeze_all()


