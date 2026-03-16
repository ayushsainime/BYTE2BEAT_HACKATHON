from __future__ import annotations

import torch

from models.multimodal_model import MultimodalRiskModel
from utils.config import FreezePolicyConfig, ModelConfig


def test_model_forward_shape() -> None:
    config = ModelConfig(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_labels=8,
        image_feature_dropout=0.2,
        metadata_hidden_dim=32,
        metadata_dropout=0.1,
        fusion_hidden_dims=[64, 32],
        fusion_dropout=0.2,
        freeze_policy=FreezePolicyConfig(enabled=True, freeze_encoder_epochs=1, unfreeze_last_n_stages=2, full_finetune_epoch=3),
    )
    model = MultimodalRiskModel(config)

    left = torch.randn(2, 3, 224, 224)
    right = torch.randn(2, 3, 224, 224)
    metadata = torch.randn(2, 2)

    logits = model(left, right, metadata)
    assert logits.shape == torch.Size([2, 8])
    assert torch.isfinite(logits).all()
