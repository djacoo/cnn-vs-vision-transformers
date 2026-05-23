"""Failure-case mosaic: most-confused breed pairs with saliency overlays."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def top_k_confused_pairs(cm, k=5):
    """Return [(true_label, pred_label, count), ...] sorted desc, diagonal excluded."""
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


def build_failure_mosaic(cnn_run, vit_run, k=5,
                         out_path="report/figures/failure_cases.png"):
    """For each of the top-k confused breed pairs, render up to 2 misclassified
    images with [original | CNN GradCAM | ViT attention] overlays."""
    from sklearn.metrics import confusion_matrix
    from src.config import Config
    from src.data import get_dataloaders
    from src.models import build_model
    from src.utils import get_device, set_seed
    from viz.gradcam import GradCAM
    from viz.attention_rollout import rollout_heatmap_for_image
    from viz.compare_figures import denormalize, overlay_heatmap

    device = get_device()

    def load(run):
        run = Path(run)
        cfg = Config(**json.loads((run / "config.json").read_text()))
        model = build_model(cfg).to(device)
        ckpt = torch.load(run / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return cfg, model

    vit_cfg, vit = load(vit_run)
    cnn_cfg, cnn = load(cnn_run)
    set_seed(vit_cfg.seed)
    loaders = get_dataloaders(vit_cfg)
    class_names = loaders["class_names"]
    gradcam = GradCAM(cnn, target_layer=cnn.layer4[-1])

    y_true, y_pred, images_buf = [], [], []
    with torch.no_grad():
        for images, targets in loaders["test"]:
            logits = vit(images.to(device))
            preds = logits.argmax(1).cpu()
            y_true.extend(targets.tolist())
            y_pred.extend(preds.tolist())
            images_buf.append(images)
    images_all = torch.cat(images_buf)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pairs = top_k_confused_pairs(cm, k=k)

    rows = []
    for true_label, pred_label, _ in pairs:
        candidates = [i for i, (t, p) in enumerate(zip(y_true, y_pred))
                      if t == true_label and p == pred_label]
        if not candidates:
            continue
        idx = candidates[0]
        rows.append((idx, true_label, pred_label, images_all[idx]))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (idx, true_label, pred_label, img) in enumerate(rows):
        img_b = img[None].to(device)
        cam = gradcam(img_b.clone(), target_class=true_label)
        att = rollout_heatmap_for_image(vit, img_b.clone(), grid_size=14)
        att = F.interpolate(att[None, None], size=(224, 224),
                            mode="bilinear", align_corners=False)[0, 0]
        axes[r, 0].imshow(denormalize(img).permute(1, 2, 0))
        axes[r, 0].set_ylabel(
            f"GT: {class_names[true_label]}\npred: {class_names[pred_label]}")
        axes[r, 1].imshow(overlay_heatmap(img, cam))
        axes[r, 2].imshow(overlay_heatmap(img, att))
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("input")
            axes[r, 1].set_title("CNN Grad-CAM")
            axes[r, 2].set_title("ViT attention")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-run", default="experiments/resnet50")
    p.add_argument("--vit-run", default="experiments/vit_b16_ft")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default="report/figures/failure_cases.png")
    args = p.parse_args()
    print(build_failure_mosaic(args.cnn_run, args.vit_run, args.k, args.out))


if __name__ == "__main__":
    main()
