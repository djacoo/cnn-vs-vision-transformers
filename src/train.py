import argparse
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config import Config, load_config
from src.data import get_dataloaders
from src.logging_utils import RunLogger
from src.models import build_model
from src.utils import count_parameters, get_device, get_logger, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += images.size(0)
    return {"loss": total_loss / seen, "acc": correct / seen}


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += images.size(0)
    return {"loss": total_loss / seen, "acc": correct / seen}


def _build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    warmup_iters = max(1, cfg.warmup_epochs * steps_per_epoch)
    total_iters = cfg.epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_iters - warmup_iters))
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_iters])


def train(cfg: Config) -> dict:
    """Full training run for one variant. Writes artifacts to experiments/<name>/."""
    logger = get_logger("train")
    set_seed(cfg.seed)
    device = get_device()
    logger.info("device=%s variant=%s", device, cfg.name)

    loaders = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg, len(loaders["train"]))

    run = RunLogger(cfg.experiments_dir, cfg.name)
    run.save_json("config.json", cfg.to_dict())

    history, best_acc, best_epoch, epochs_no_improve = [], 0.0, -1, 0
    start = time.time()
    for epoch in tqdm(range(cfg.epochs), desc=cfg.name):
        tr = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scheduler)
        va = evaluate_model(model, loaders["val"], criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        run.log_scalars(epoch, train_loss=tr["loss"], train_acc=tr["acc"],
                        val_loss=va["loss"], val_acc=va["acc"], lr=lr_now)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "lr": lr_now})

        if va["acc"] > best_acc:
            best_acc, best_epoch, epochs_no_improve = va["acc"], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(),
                        "epoch": epoch, "val_acc": best_acc},
                       run.run_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    elapsed = time.time() - start
    run.save_json("history.json", {"history": history})
    run.save_json("metadata.json", {
        "variant": cfg.name,
        "total_params": count_parameters(model),
        "trainable_params": count_parameters(model, trainable_only=True),
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "train_time_sec": elapsed,
        "epochs_run": len(history),
    })
    run.close()
    logger.info("done variant=%s best_val_acc=%.4f time=%.0fs", cfg.name, best_acc, elapsed)
    return {"best_val_acc": best_acc, "ckpt_path": str(run.run_dir / "best.pt")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
