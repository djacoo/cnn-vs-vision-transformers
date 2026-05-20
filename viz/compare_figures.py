import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.config import Config
from src.data import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from src.models import build_model
from src.utils import get_device, set_seed
from viz.gradcam import GradCAM
from viz.attention_rollout import rollout_heatmap_for_image
import json


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization; returns a (3,H,W) tensor in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1)


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
    """Blend a (H,W) heatmap over a (3,H,W) image. Returns (H,W,3) in [0,1]."""
    rgb = denormalize(image).permute(1, 2, 0)
    colored = torch.from_numpy(cm.jet(heatmap.numpy())[..., :3]).float()
    return (alpha * colored + (1 - alpha) * rgb).clamp(0, 1)


def _load_variant(run_dir: str, device):
    run = Path(run_dir)
    cfg = Config(**json.loads((run / "config.json").read_text()))
    model = build_model(cfg).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cfg, model


def build_comparison_figure(cnn_run: str, vit_run: str, n_images: int = 6,
                            out_path: str = "report/figures/comparison.png") -> str:
    """Render rows of [original | CNN Grad-CAM | ViT attention] (spec §8)."""
    device = get_device()
    cnn_cfg, cnn = _load_variant(cnn_run, device)
    _, vit = _load_variant(vit_run, device)
    set_seed(cnn_cfg.seed)

    loaders = get_dataloaders(cnn_cfg)
    images, _ = next(iter(loaders["test"]))
    images = images[:n_images]

    gradcam = GradCAM(cnn, target_layer=cnn.layer4[-1])

    fig, axes = plt.subplots(n_images, 3, figsize=(9, 3 * n_images))
    cols = ["original", "CNN Grad-CAM", "ViT attention"]
    for row in range(n_images):
        img = images[row:row + 1].to(device)
        cam = gradcam(img.clone())
        att = rollout_heatmap_for_image(vit, img.clone(), grid_size=14)
        att = F.interpolate(att[None, None], size=(224, 224),
                            mode="bilinear", align_corners=False)[0, 0]

        axes[row, 0].imshow(denormalize(images[row]).permute(1, 2, 0))
        axes[row, 1].imshow(overlay_heatmap(images[row], cam))
        axes[row, 2].imshow(overlay_heatmap(images[row], att))
        for col in range(3):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(cols[col])

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-run", default="experiments/resnet50")
    parser.add_argument("--vit-run", default="experiments/vit_b16_ft")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--out", default="report/figures/comparison.png")
    args = parser.parse_args()
    print(build_comparison_figure(args.cnn_run, args.vit_run, args.n_images, args.out))


if __name__ == "__main__":
    main()
