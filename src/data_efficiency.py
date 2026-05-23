"""Train each variant at multiple train_fractions; aggregate results into a CSV."""
import argparse
import csv
import json
from pathlib import Path

from src.config import load_config
from src.train import train
from src.evaluate import evaluate_variant


VARIANTS = ["resnet50", "vit_b16_ft", "vit_s16_scratch",
            "deit_s16_ft", "vit_b16_linprobe"]
FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def _run_one(config_name, fraction, base_experiments_dir="experiments_data_eff"):
    cfg = load_config(f"configs/{config_name}.yaml")
    cfg.train_fraction = fraction
    cfg.name = f"{config_name}_f{int(fraction*100):03d}"
    cfg.experiments_dir = base_experiments_dir
    cfg.validate()
    train(cfg)
    metrics = evaluate_variant(f"{base_experiments_dir}/{cfg.name}")
    return {"variant": config_name, "train_fraction": fraction, **metrics}


def sweep(variants=None, fractions=None, out_csv="experiments_data_eff/results.csv"):
    variants = variants or VARIANTS
    fractions = fractions or FRACTIONS
    rows = []
    for v in variants:
        for f in fractions:
            rows.append(_run_one(v, f))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    p.add_argument("--fractions", nargs="*", type=float, default=FRACTIONS)
    p.add_argument("--out", default="experiments_data_eff/results.csv")
    args = p.parse_args()
    sweep(args.variants, args.fractions, args.out)


if __name__ == "__main__":
    main()
