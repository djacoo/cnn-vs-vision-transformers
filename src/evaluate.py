import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)

from src.config import Config
from src.data import get_dataloaders
from src.models import build_model
from src.utils import get_device, get_logger, set_seed


def compute_metrics(y_true, y_pred) -> dict:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_f1": float(f1)}


@torch.no_grad()
def _collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(targets.tolist())
    return y_true, y_pred


def _plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation="vertical", colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_curves(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_variant(run_dir: str) -> dict:
    """Evaluate a finished run on the test split; write metrics + figures."""
    logger = get_logger("evaluate")
    run = Path(run_dir)
    cfg = Config(**json.loads((run / "config.json").read_text()))
    set_seed(cfg.seed)
    device = get_device()

    model = build_model(cfg).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    loaders = get_dataloaders(cfg)
    y_true, y_pred = _collect_predictions(model, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred)

    _plot_confusion_matrix(y_true, y_pred, loaders["class_names"],
                           run / "confusion_matrix.png")
    history = json.loads((run / "history.json").read_text())["history"]
    _plot_curves(history, run / "curves.png")
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("variant=%s test_acc=%.4f macro_f1=%.4f",
                cfg.name, metrics["accuracy"], metrics["macro_f1"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    evaluate_variant(args.run_dir)


if __name__ == "__main__":
    main()
