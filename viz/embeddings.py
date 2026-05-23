"""Extract penultimate features from a trained variant and plot 2D t-SNE."""
import json
from pathlib import Path

import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def extract_features(model, images, layer, device):
    """Run `model` and return penultimate features.

    For torchvision ResNets pass `layer=model.avgpool`; for timm ViTs pass `None`
    and we call `model.forward_features` and read the CLS token. Otherwise the
    full forward result is returned.
    """
    model.eval()
    images = images.to(device)
    if layer is None:
        if hasattr(model, "forward_features"):
            feats = model.forward_features(images)
            if feats.dim() == 3:  # ViT (B, tokens, D)
                feats = feats[:, 0]  # CLS token
            return feats.cpu()
        return model(images).cpu()
    holder = {}

    def hook(_m, _i, o):
        holder["out"] = o

    h = layer.register_forward_hook(hook)
    try:
        model(images)
    finally:
        h.remove()
    feats = holder["out"]
    if feats.dim() > 2:
        feats = feats.flatten(1)
    return feats.cpu()


def collect_features(run_dir, max_images=600):
    """Walk the val loader, return (features, labels, class_names)."""
    from src.config import Config
    from src.data import get_dataloaders
    from src.models import build_model
    from src.utils import get_device, set_seed

    run = Path(run_dir)
    cfg = Config(**json.loads((run / "config.json").read_text()))
    set_seed(cfg.seed)
    device = get_device()
    model = build_model(cfg).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    layer = model.avgpool if cfg.backbone == "resnet50" else None
    loaders = get_dataloaders(cfg)
    feats, labels = [], []
    for images, targets in tqdm(loaders["val"], desc=f"feats {cfg.name}"):
        feats.append(extract_features(model, images, layer, device))
        labels.append(targets)
        if sum(t.size(0) for t in labels) >= max_images:
            break
    return torch.cat(feats)[:max_images], torch.cat(labels)[:max_images], loaders["class_names"]


def tsne_plot_panel(run_dirs, out_path="report/figures/tsne_embeddings.png",
                    max_images=600, perplexity=30, seed=42):
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, len(run_dirs), figsize=(5 * len(run_dirs), 5))
    if len(run_dirs) == 1:
        axes = [axes]
    for ax, run in zip(axes, run_dirs):
        feats, labels, _ = collect_features(run, max_images=max_images)
        proj = TSNE(n_components=2, perplexity=perplexity,
                    init="pca", random_state=seed).fit_transform(feats.numpy())
        ax.scatter(proj[:, 0], proj[:, 1], c=labels.numpy(), cmap="tab20",
                   s=8, alpha=0.7)
        ax.set_title(Path(run).name)
        ax.set_xticks([]); ax.set_yticks([])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
