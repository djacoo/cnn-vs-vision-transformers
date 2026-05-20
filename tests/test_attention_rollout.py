import timm
import torch

from viz.attention_rollout import attention_rollout, ViTAttentionExtractor


def test_attention_rollout_pure_math_shape():
    # 2 layers, 1 head, 5 tokens (1 CLS + 4 patches)
    attns = [torch.rand(1, 1, 5, 5).softmax(-1) for _ in range(2)]
    heat = attention_rollout(attns, grid_size=2)
    assert heat.shape == (2, 2)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0


def test_extractor_collects_one_map_per_block():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=37)
    extractor = ViTAttentionExtractor(model)
    attns = extractor(torch.randn(1, 3, 224, 224))
    assert len(attns) == len(model.blocks)
    # ViT-S/16 @224: 196 patches + CLS = 197 tokens
    assert attns[0].shape[-1] == 197


def test_rollout_from_extractor_produces_14x14():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=37)
    extractor = ViTAttentionExtractor(model)
    attns = extractor(torch.randn(1, 3, 224, 224))
    heat = attention_rollout(attns, grid_size=14)
    assert heat.shape == (14, 14)
