import pytest
import torch

from viz.compare_figures import denormalize, overlay_heatmap


def test_denormalize_inverts_imagenet_normalization():
    from src.data import IMAGENET_MEAN, IMAGENET_STD
    from torchvision import transforms
    raw = torch.rand(3, 224, 224)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(raw)
    recovered = denormalize(norm)
    assert torch.allclose(recovered, raw, atol=1e-5)
    assert recovered.min() >= 0.0 and recovered.max() <= 1.0


def test_overlay_heatmap_returns_rgb_image():
    image = torch.rand(3, 224, 224)
    heat = torch.rand(224, 224)
    blended = overlay_heatmap(image, heat)
    assert blended.shape == (224, 224, 3)


@pytest.mark.slow
def test_build_multi_method_figure_produces_file(tmp_path):
    """Integration test: requires real checkpoints in experiments/."""
    from viz.compare_figures import build_multi_method_figure
    out = tmp_path / "multi_method_comparison.png"
    result = build_multi_method_figure(
        cnn_run="experiments/resnet50",
        vit_runs=["experiments/vit_b16_ft", "experiments/vit_b16_linprobe",
                  "experiments/vit_s16_dino_linprobe"],
        n_images=2,
        out_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert result == str(out)
