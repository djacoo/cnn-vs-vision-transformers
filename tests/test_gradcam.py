import torch
import torchvision

from viz.gradcam import GradCAM


def test_gradcam_heatmap_shape_and_range():
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model.eval()
    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam(torch.randn(1, 3, 224, 224))
    assert heatmap.shape == (224, 224)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0


def test_gradcam_respects_explicit_target_class():
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model.eval()
    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam(torch.randn(1, 3, 224, 224), target_class=5)
    assert heatmap.shape == (224, 224)
