import timm
import torch.nn as nn
import torchvision

from src.config import Config


def build_model(cfg: Config) -> nn.Module:
    """Build a backbone with a 37-class head behind one interface."""
    if cfg.backbone == "resnet50":
        weights = "IMAGENET1K_V2" if cfg.pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    else:  # timm ViT / DeiT
        model = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=cfg.num_classes
        )

    if cfg.protocol == "linear_probe":
        _freeze_backbone(model, cfg.backbone)
    return model


def _freeze_backbone(model: nn.Module, backbone: str) -> None:
    """Freeze every parameter, then re-enable the classifier head only."""
    for param in model.parameters():
        param.requires_grad = False
    head = model.fc if backbone == "resnet50" else model.get_classifier()
    for param in head.parameters():
        param.requires_grad = True


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    """Return the last conv block of a ResNet for Grad-CAM."""
    return model.layer4[-1]
