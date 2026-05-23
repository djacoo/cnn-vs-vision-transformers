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

## Extensions

Beyond the five-variant baseline, this branch adds modules for additional analyses (built and tested; runtime execution is deferred):

### Sixth variant (configured, not yet trained)
| # | Config | Backbone | Pretraining | Protocol | Role |
|---|---|---|---|---|---|
| 6 | `vit_s16_dino_linprobe` | ViT-S/16 | DINO (self-supervised) | linear probe | self-supervised pretraining comparison |

Train with:
```bash
python -m src.train --config configs/vit_s16_dino_linprobe.yaml
python -m src.evaluate --run-dir experiments/vit_s16_dino_linprobe
```

### Analyses

- **CLIP zero-shot baseline** (`src/clip_zeroshot.py`): vision-language ViT-B/32 with prompt ensembling, no training.
- **DINO self-attention visualizer** (`viz/dino_attention.py`): per-head + mean attention maps from the last transformer block.
- **Quantitative saliency metrics** (`src/saliency_metrics.py`): Pointing Game, Bbox-IoU @ top-k, Deletion/Insertion AUC against Oxford-IIIT Pet head bounding boxes (`src/pet_bboxes.py`).
- **Data-efficiency sweep** (`src/data_efficiency.py`): trains each supervised variant at 10/25/50/100% train fraction; writes a results CSV.
- **t-SNE feature embeddings** (`viz/embeddings.py`): 2D projection of penultimate features across variants, colored by class.
- **Failure-case mosaic** (`viz/failures.py`): top-K confused breed pairs with CNN Grad-CAM + ViT attention overlays.

### Reproducing each analysis

```bash
# CLIP zero-shot (~5-10 min, requires `pip install -r requirements.txt`)
python -m src.clip_zeroshot

# Saliency metrics per trained variant (~5 min each)
for v in resnet50 vit_b16_ft deit_s16_ft vit_b16_linprobe vit_s16_dino_linprobe; do
  python -m src.saliency_metrics --run-dir experiments/$v --max-images 200
done

# Data-efficiency sweep (~3-5 h, all 5 supervised variants × 4 train fractions)
python -m src.data_efficiency

# t-SNE panel across selected variants
python -c "from viz.embeddings import tsne_plot_panel; tsne_plot_panel([\
'experiments/resnet50','experiments/vit_b16_ft',\
'experiments/vit_s16_scratch','experiments/vit_s16_dino_linprobe'])"

# Failure-case mosaic
python -m viz.failures

# DINO per-head attention (requires vit_s16_dino_linprobe trained)
python -c "
import torch, json
from pathlib import Path
from src.config import Config
from src.data import get_dataloaders
from src.models import build_model
from src.utils import get_device, set_seed
from viz.dino_attention import render_dino_figure
run = Path('experiments/vit_s16_dino_linprobe')
cfg = Config(**json.loads((run / 'config.json').read_text()))
set_seed(cfg.seed)
device = get_device()
model = build_model(cfg).to(device)
ckpt = torch.load(run / 'best.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
loaders = get_dataloaders(cfg)
images, _ = next(iter(loaders['test']))
render_dino_figure(model.to(device), images[:6].to(device),
                   'report/figures/dino_attention.png')
"
```

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
