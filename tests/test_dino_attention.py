import pytest
import torch
import timm

from viz.dino_attention import last_block_attention_heads


@pytest.mark.slow
def test_returns_one_heatmap_per_head():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=37)
    image = torch.randn(1, 3, 224, 224)
    heads = last_block_attention_heads(model, image, grid_size=14)
    # ViT-S/16 has 6 heads
    assert heads.shape == (6, 14, 14)
    assert heads.min() >= 0.0
    assert heads.max() <= 1.0 + 1e-6
