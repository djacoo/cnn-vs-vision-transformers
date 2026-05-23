"""Quantitative saliency metrics: Pointing Game, Bbox-IoU, Deletion/Insertion AUC."""
import torch
import torch.nn.functional as F


def pointing_game_hit(heatmap, bbox_mask):
    """Argmax of heatmap inside bbox_mask?"""
    flat = heatmap.flatten().argmax().item()
    h, w = heatmap.shape
    y, x = divmod(flat, w)
    return bool(bbox_mask[y, x].item())


def bbox_iou_at_threshold(heatmap, bbox_mask, top_fraction=0.2):
    """IoU between the top-`top_fraction` pixels of `heatmap` and `bbox_mask`."""
    flat = heatmap.flatten()
    k = max(1, int(top_fraction * flat.numel()))
    top_idx = flat.topk(k).indices
    sal_mask = torch.zeros_like(flat, dtype=torch.bool)
    sal_mask[top_idx] = True
    sal_mask = sal_mask.view_as(heatmap)
    inter = (sal_mask & bbox_mask).sum().item()
    union = (sal_mask | bbox_mask).sum().item()
    return inter / union if union > 0 else 0.0


def _ranked_pixel_indices(heatmap):
    """Pixel coordinates sorted by saliency, most to least salient."""
    flat = heatmap.flatten()
    order = flat.argsort(descending=True)
    h, w = heatmap.shape
    ys = order // w
    xs = order % w
    return ys, xs


@torch.no_grad()
def deletion_auc(model, image, heatmap, target_class, device,
                 n_steps=20, replace_value=0.0):
    """Drop the most-salient pixels in `n_steps` chunks; return mean prob over the curve."""
    ys, xs = _ranked_pixel_indices(heatmap)
    ys, xs = ys.to(device), xs.to(device)
    total = ys.numel()
    chunk = max(1, total // n_steps)
    img = image.clone().to(device)
    probs = []
    for step in range(n_steps + 1):
        logits = model(img)
        prob = F.softmax(logits, dim=1)[0, target_class].item()
        probs.append(prob)
        start = step * chunk
        end = min(total, start + chunk)
        sel_y, sel_x = ys[start:end], xs[start:end]
        img[0, :, sel_y, sel_x] = replace_value
    return sum(probs) / len(probs)


@torch.no_grad()
def insertion_auc(model, image, heatmap, target_class, device,
                  n_steps=20, blur_sigma=15.0):
    """Reveal most-salient pixels on a blurred base; return mean prob over the curve."""
    from torchvision.transforms.functional import gaussian_blur

    ys, xs = _ranked_pixel_indices(heatmap)
    ys, xs = ys.to(device), xs.to(device)
    total = ys.numel()
    chunk = max(1, total // n_steps)
    target_img = image.to(device)
    base = gaussian_blur(target_img, kernel_size=51, sigma=blur_sigma)
    img = base.clone()
    probs = []
    for step in range(n_steps + 1):
        logits = model(img)
        prob = F.softmax(logits, dim=1)[0, target_class].item()
        probs.append(prob)
        start = step * chunk
        end = min(total, start + chunk)
        sel_y, sel_x = ys[start:end], xs[start:end]
        img[0, :, sel_y, sel_x] = target_img[0, :, sel_y, sel_x]
    return sum(probs) / len(probs)


def evaluate_saliency_for_run(run_dir, max_images=200, top_fraction=0.2,
                              out_path=None):
    """Aggregate saliency metrics for a finished run on its val set.

    Picks the right saliency producer per backbone:
      - resnet50  -> Grad-CAM on layer4[-1]
      - ViT/DeiT  -> attention rollout

    Writes one JSON to `<run_dir>/saliency_metrics.json`.
    """
    import json
    from pathlib import Path

    import torch
    from torchvision.datasets import OxfordIIITPet

    from src.config import Config
    from src.data import build_transforms, stratified_split
    from src.models import build_model
    from src.utils import get_device, set_seed
    from src.pet_bboxes import load_all_bboxes, bbox_to_resized_mask
    from viz.gradcam import GradCAM
    from viz.attention_rollout import rollout_heatmap_for_image

    run = Path(run_dir)
    cfg = Config(**json.loads((run / "config.json").read_text()))
    set_seed(cfg.seed)
    device = get_device()
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    eval_tf = build_transforms(cfg, train=False)
    trainval = OxfordIIITPet(cfg.data_root, split="trainval",
                             target_types="category", transform=eval_tf,
                             download=False)
    labels = list(trainval._labels)
    _, val_idx = stratified_split(labels, cfg.val_fraction, cfg.seed)
    bboxes = load_all_bboxes(cfg.data_root)

    is_resnet = cfg.backbone == "resnet50"
    if is_resnet:
        gradcam = GradCAM(model, target_layer=model.layer4[-1])

    point_hits, ious, deletions, insertions = [], [], [], []
    seen = 0
    image_files = trainval._images
    for idx in val_idx:
        if seen >= max_images:
            break
        stem = Path(image_files[idx]).stem
        if stem not in bboxes:
            continue
        image, target = trainval[idx]
        img = image[None].to(device)

        if is_resnet:
            heat = gradcam(img.clone(), target_class=int(target))
            heat224 = heat
        else:
            heat = rollout_heatmap_for_image(model, img.clone(), grid_size=14)
            heat224 = torch.nn.functional.interpolate(
                heat[None, None], size=(224, 224),
                mode="bilinear", align_corners=False)[0, 0]

        bbox_mask = bbox_to_resized_mask(bboxes[stem], target_size=224)
        point_hits.append(int(pointing_game_hit(heat224.cpu(), bbox_mask)))
        ious.append(bbox_iou_at_threshold(heat224.cpu(), bbox_mask, top_fraction))
        deletions.append(deletion_auc(model, img, heat224, int(target),
                                      device, n_steps=20))
        insertions.append(insertion_auc(model, img, heat224, int(target),
                                        device, n_steps=20))
        seen += 1

    metrics = {
        "n_images": seen,
        "pointing_game": sum(point_hits) / seen if seen else 0.0,
        "bbox_iou_at_top20": sum(ious) / seen if seen else 0.0,
        "deletion_auc": sum(deletions) / seen if seen else 0.0,
        "insertion_auc": sum(insertions) / seen if seen else 0.0,
    }
    out_path = out_path or (run / "saliency_metrics.json")
    Path(out_path).write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--max-images", type=int, default=200)
    args = p.parse_args()
    print(evaluate_saliency_for_run(args.run_dir, max_images=args.max_images))


if __name__ == "__main__":
    main()
