import torch

from src.config import Config
from src.models import build_model, get_gradcam_layer
from src.utils import count_parameters


def _cfg(backbone, protocol, **ov):
    base = dict(name="m", backbone=backbone, pretrained=False,
                protocol=protocol, augmentation="light", lr=1e-4, epochs=1)
    base.update(ov)
    return Config(**base)  # not validated: pretrained=False on purpose


def test_resnet_head_has_37_outputs():
    model = build_model(_cfg("resnet50", "full_ft"))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)


def test_vit_head_has_37_outputs():
    model = build_model(_cfg("vit_small_patch16_224", "scratch"))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)


def test_linear_probe_freezes_backbone_only():
    model = build_model(_cfg("vit_base_patch16_224", "linear_probe"))
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    assert 0 < trainable < total
    # trainable params are exactly the classifier head
    assert trainable == 37 * 768 + 37  # ViT-B hidden dim 768


def test_full_ft_keeps_all_params_trainable():
    model = build_model(_cfg("resnet50", "full_ft"))
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_get_gradcam_layer_returns_module():
    model = build_model(_cfg("resnet50", "full_ft"))
    layer = get_gradcam_layer(model)
    assert isinstance(layer, torch.nn.Module)
