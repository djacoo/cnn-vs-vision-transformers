"""CLIP zero-shot classification on Oxford-IIIT Pets (no training)."""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils import get_device, get_logger, set_seed

DEFAULT_TEMPLATES = [
    "a photo of a {}, a type of pet.",
    "a photo of a {}.",
    "a picture of a {} pet.",
    "an image of a {} cat or dog.",
]


def build_prompts(class_names, templates=None):
    """Cartesian product of templates x class_names with names lowercased
    and underscores replaced by spaces."""
    templates = templates or DEFAULT_TEMPLATES
    prompts = []
    for t in templates:
        for name in class_names:
            clean = name.replace("_", " ").lower()
            prompts.append(t.format(clean))
    return prompts


def _build_text_features(model, tokenizer, class_names, templates, device):
    """Encode N_templates*N_classes text prompts, average per class, normalize."""
    n_classes = len(class_names)
    prompts = build_prompts(class_names, templates)
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text = model.encode_text(tokens)
    text = text / text.norm(dim=-1, keepdim=True)
    text = text.reshape(len(templates), n_classes, -1).mean(0)
    text = text / text.norm(dim=-1, keepdim=True)
    return text


@torch.no_grad()
def zero_shot_evaluate(model_name="ViT-B-32-quickgelu", pretrained="openai",
                       templates=None, out_dir="experiments/clip_zeroshot",
                       data_root="data", batch_size=32, num_workers=4,
                       seed=42):
    """Run zero-shot inference on the Pet37 test split. Writes test_metrics.json.

    Uses CLIP's own image preprocess (not ImageNet normalization) so the
    reported accuracy reflects CLIP under its canonical input distribution.
    """
    import open_clip
    from torch.utils.data import DataLoader
    from torchvision.datasets import OxfordIIITPet

    logger = get_logger("clip_zeroshot")
    device = get_device()
    set_seed(seed)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device).eval()

    test_ds = OxfordIIITPet(data_root, split="test", target_types="category",
                            transform=preprocess, download=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             persistent_workers=num_workers > 0)
    class_names = list(test_ds.classes)
    templates = templates or DEFAULT_TEMPLATES
    text_features = _build_text_features(model, tokenizer, class_names,
                                         templates, device)

    correct = total = 0
    for images, targets in tqdm(test_loader, desc="clip zero-shot"):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        preds = logits.argmax(1).cpu()
        correct += (preds == targets).sum().item()
        total += targets.numel()

    accuracy = correct / total
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"accuracy": accuracy, "n_test": total, "model": model_name,
               "pretrained": pretrained, "n_templates": len(templates),
               "preprocess": "clip"}
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("clip zero-shot test_acc=%.4f templates=%d n_test=%d",
                accuracy, len(templates), total)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--out-dir", default="experiments/clip_zeroshot")
    args = parser.parse_args()
    zero_shot_evaluate(args.model, args.pretrained, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
