# CNN vs Vision Transformer — Where Does the Model Look?

A comparative study of CNN and ViT image classifiers on the Oxford-IIIT Pets fine-grained dataset, focused not only on which model is more accurate but on *where* each model looks — visualized with Grad-CAM (CNN) and attention rollout (ViT).

## The Five Variants

| # | Config | Backbone | Pretrained | Protocol | Role |
|---|---|---|---|---|---|
| 1 | `resnet50` | ResNet-50 | ImageNet | full fine-tune | CNN baseline |
| 2 | `vit_b16_ft` | ViT-B/16 | ImageNet-21k→1k | full fine-tune | Transformer + transfer learning |
| 3 | `vit_s16_scratch` | ViT-S/16 | none | from scratch | Isolates pretraining (data-hunger) |
| 4 | `deit_s16_ft` | DeiT-S/16 | ImageNet | full fine-tune | Data-efficient transformer |
| 5 | `vit_b16_linprobe` | ViT-B/16 | ImageNet | linear probe (frozen backbone) | Protocol study vs #2 |

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv     # macOS/Homebrew path; adjust for your system
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
# Train a variant (downloads Oxford-IIIT Pets on first run)
python -m src.train --config configs/resnet50.yaml

# Evaluate a finished run on the test split
python -m src.evaluate --run-dir experiments/resnet50

# Generate the side-by-side saliency figure
python -m viz.compare_figures --cnn-run experiments/resnet50 --vit-run experiments/vit_b16_ft
```

The five config names available under `configs/` are: `resnet50`, `vit_b16_ft`, `vit_s16_scratch`, `deit_s16_ft`, `vit_b16_linprobe`. Training uses the PyTorch MPS (Metal) backend on Apple Silicon with a CPU fallback on other systems. Runs log to TensorBoard:

```bash
tensorboard --logdir experiments
```

## Results

Test-set performance on the 37-class Oxford-IIIT Pets dataset (official test split):

| Variant | Test acc | Macro F1 | Total params | Train time |
|---|---|---|---|---|
| `resnet50` | 0.925 | 0.924 | 23.6M | 6.3 min |
| `vit_b16_ft` | 0.937 | 0.936 | 85.8M | 21.4 min |
| `vit_s16_scratch` | 0.227 | 0.221 | 21.7M | 19.5 min |
| `deit_s16_ft` | 0.927 | 0.926 | 21.7M | 5.8 min |
| `vit_b16_linprobe` | 0.924 | 0.923 | 85.8M (28k trainable) | 5.2 min |

The three fine-tuned/transfer variants (ResNet-50, ViT-B/16 FT, DeiT-S/16 FT) land within ~1 pp of each other; the from-scratch ViT collapses to ~0.23, the canonical data-hunger result.

![Saliency comparison](report/figures/comparison.png)

`report/figures/` also holds per-variant confusion matrices, training curves, the accuracy/cost trade-off plot, and `results_table.csv`.

## Repository Layout

```
configs/          # one YAML per variant
src/              # data loading, models, training, evaluation
viz/              # Grad-CAM, attention rollout, comparison figure generator
tests/            # unit and integration tests
notebooks/        # results_analysis.ipynb
experiments/      # run artifacts — gitignored
report/figures/   # committed figures and results_table.csv
docs/             # project description and implementation plan
```

## Tests

```bash
pytest              # fast suite (~30 tests)
pytest -m slow      # additionally runs the dataset-download integration test
```

## License

MIT — see `LICENSE`.
