# CNN vs Vision Transformer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch pipeline that trains 5 CNN/ViT variants on Oxford-IIIT Pets, evaluates them, and visualizes where each model looks (Grad-CAM vs attention rollout).

**Architecture:** A config-driven pipeline. One YAML per variant feeds a `Config` dataclass. A single model factory builds ResNet/ViT/DeiT behind one interface, so one training loop and one evaluator serve every variant. Visualization modules attach hooks to a trained model to produce saliency maps. All runs log to TensorBoard and write self-contained artifact folders under `experiments/`.

**Tech Stack:** Python 3.12, PyTorch (MPS backend), torchvision, timm, scikit-learn, matplotlib, pyyaml, tensorboard, pytest.

---

## Conventions

- **Working directory:** repo root. All commands run from there.
- **Run modules as packages:** `python -m src.train --config configs/resnet50.yaml`.
- **Git Flow:** one `feature/<n>-<name>` branch per task off `develop`; merge back with `--no-ff`; the `<n>` matches the spec issue number in `docs/project-description.md` §12.
- **Tests:** `pytest` from repo root. Pure logic gets real unit tests (TDD). Integration-heavy code (training loop, full runs) gets synthetic smoke tests; tests needing the real dataset are marked `@pytest.mark.slow` and skipped by default.
- **Device:** always resolve through `get_device()`; never hardcode `mps`/`cpu`.
- **Determinism:** every entry point calls `set_seed(cfg.seed)` first.

## File Structure

| Path | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Exclude `experiments/`, `data/`, `.venv/`, caches |
| `README.md` | Setup + reproduction instructions |
| `pytest.ini` | Test config, registers `slow` marker |
| `src/__init__.py` | Marks `src` a package |
| `src/utils.py` | `get_device`, `set_seed`, `get_logger`, `count_parameters` |
| `src/logging_utils.py` | `RunLogger` — TensorBoard wrapper + run-dir management |
| `src/config.py` | `Config` dataclass, `load_config`, validation |
| `src/data.py` | `stratified_split`, `build_transforms`, `get_dataloaders` |
| `src/models.py` | `build_model` factory, `get_gradcam_layer` helper |
| `src/train.py` | `train_one_epoch`, `evaluate_model`, `train`, CLI |
| `src/evaluate.py` | `compute_metrics`, confusion matrix + curves, CLI |
| `configs/*.yaml` | One config per variant (5 files) |
| `viz/__init__.py` | Marks `viz` a package |
| `viz/gradcam.py` | `GradCAM` class for the CNN |
| `viz/attention_rollout.py` | `attention_rollout` fn + `ViTAttentionExtractor` |
| `viz/compare_figures.py` | `original | Grad-CAM | attention` figure generator |
| `tests/*.py` | Unit + smoke tests mirroring `src/` and `viz/` |
| `notebooks/results_analysis.ipynb` | Final aggregation / analysis notebook |
| `experiments/` | gitignored: checkpoints, TB logs, metrics, figures |
| `report/figures/` | Final exported figures for report/slides |

---

## Task 1: Project scaffold & environment (issue #1)

**Branch:** `feature/1-repo-init`

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pytest.ini`, `README.md`
- Create: `src/__init__.py`, `viz/__init__.py`, `tests/__init__.py`
- Create dirs: `configs/`, `notebooks/`, `experiments/.gitkeep`, `report/figures/.gitkeep`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git checkout -b feature/1-repo-init develop
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/

# Project artifacts (large — never commit)
experiments/*
!experiments/.gitkeep
data/
report/figures/*
!report/figures/.gitkeep

# OS / editor
.DS_Store
.ipynb_checkpoints/
```

- [ ] **Step 3: Write `requirements.txt`**

```
torch>=2.3
torchvision>=0.18
timm>=1.0.7
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
pyyaml>=6.0
tensorboard>=2.16
tqdm>=4.66
pytest>=8.0
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    slow: tests that download the dataset or train models (deselected by default)
addopts = -m "not slow"
```

- [ ] **Step 5: Create package markers and directory keepers**

```bash
touch src/__init__.py viz/__init__.py tests/__init__.py
mkdir -p configs notebooks experiments report/figures
touch experiments/.gitkeep report/figures/.gitkeep
```

- [ ] **Step 6: Write a minimal `README.md`**

Content: project title + one-line description, a "Setup" section with the venv commands below, and a "Status: under construction" note. (Full README is finished in Task 15.)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 7: Create the venv and install dependencies**

Run:
```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Expected: all packages install without error (download is a few hundred MB).

- [ ] **Step 8: Verify the environment and MPS**

Run:
```bash
source .venv/bin/activate
python -c "import torch, torchvision, timm, sklearn; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
```
Expected: prints a torch version and `mps True`.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt pytest.ini README.md src/__init__.py viz/__init__.py tests/__init__.py experiments/.gitkeep report/figures/.gitkeep
git commit -m "chore: project scaffold, dependencies, venv setup"
```

---

## Task 2: Device, seed & logging utilities (issue #1)

**Branch:** `feature/1-repo-init` (continues)

**Files:**
- Create: `src/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils.py
import torch
from src.utils import get_device, set_seed, count_parameters


def test_get_device_returns_valid_device():
    dev = get_device()
    assert dev.type in {"mps", "cuda", "cpu"}


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_count_parameters_trainable_vs_total():
    model = torch.nn.Linear(10, 4)
    total = count_parameters(model, trainable_only=False)
    assert total == 10 * 4 + 4
    for p in model.parameters():
        p.requires_grad = False
    assert count_parameters(model, trainable_only=True) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils'`.

- [ ] **Step 3: Implement `src/utils.py`**

```python
import logging
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed all RNGs used in the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: device, seed and parameter-count utilities"
```

---

## Task 3: TensorBoard run logger (issue #2)

**Branch:** `feature/2-config-logging`

**Files:**
- Create: `src/logging_utils.py`
- Test: `tests/test_logging_utils.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/1-repo-init -m "Merge feature/1-repo-init into develop"
git branch -d feature/1-repo-init
git checkout -b feature/2-config-logging develop
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_logging_utils.py
import json

from src.logging_utils import RunLogger


def test_run_logger_creates_run_dir_and_writes(tmp_path):
    logger = RunLogger(experiments_dir=str(tmp_path), run_name="demo")
    assert logger.run_dir.exists()
    logger.log_scalars(step=0, train_loss=1.0, val_acc=0.5)
    logger.save_json("metadata.json", {"params": 42})
    logger.close()
    saved = json.loads((logger.run_dir / "metadata.json").read_text())
    assert saved["params"] == 42
    assert (logger.run_dir / "tb").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_logging_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.logging_utils'`.

- [ ] **Step 4: Implement `src/logging_utils.py`**

```python
import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    """Owns a single run directory: TensorBoard logs + JSON artifacts."""

    def __init__(self, experiments_dir: str, run_name: str):
        self.run_dir = Path(experiments_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def log_scalars(self, step: int, **scalars: float) -> None:
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)

    def log_figure(self, tag: str, figure, step: int = 0) -> None:
        self.writer.add_figure(tag, figure, step)

    def save_json(self, filename: str, payload: dict[str, Any]) -> None:
        (self.run_dir / filename).write_text(json.dumps(payload, indent=2))

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_logging_utils.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/logging_utils.py tests/test_logging_utils.py
git commit -m "feat: TensorBoard run logger with run-dir management"
```

---

## Task 4: Config dataclass & YAML loader (issue #2)

**Branch:** `feature/2-config-logging` (continues)

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest

from src.config import Config, load_config


def _valid_kwargs(**overrides):
    base = dict(
        name="demo", backbone="resnet50", pretrained=True,
        protocol="full_ft", augmentation="light", lr=1e-4, epochs=20,
    )
    base.update(overrides)
    return base


def test_config_defaults():
    cfg = Config(**_valid_kwargs())
    assert cfg.num_classes == 37
    assert cfg.img_size == 224
    assert cfg.batch_size == 32
    assert cfg.weight_decay == 0.05
    assert cfg.label_smoothing == 0.1
    assert cfg.seed == 42


def test_config_rejects_unknown_protocol():
    with pytest.raises(ValueError):
        Config(**_valid_kwargs(protocol="bogus")).validate()


def test_config_rejects_pretrained_scratch_mismatch():
    # scratch protocol must not use pretrained weights
    with pytest.raises(ValueError):
        Config(**_valid_kwargs(protocol="scratch", pretrained=True)).validate()
    # fine-tune / linear probe require pretrained weights
    with pytest.raises(ValueError):
        Config(**_valid_kwargs(protocol="full_ft", pretrained=False)).validate()


def test_load_config_roundtrip(tmp_path):
    yaml_text = (
        "name: t\nbackbone: resnet50\npretrained: true\n"
        "protocol: full_ft\naugmentation: light\nlr: 0.0001\nepochs: 3\n"
    )
    path = tmp_path / "t.yaml"
    path.write_text(yaml_text)
    cfg = load_config(str(path))
    assert cfg.name == "t" and cfg.epochs == 3 and cfg.protocol == "full_ft"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`.

- [ ] **Step 3: Implement `src/config.py`**

```python
from dataclasses import asdict, dataclass

import yaml

PROTOCOLS = {"full_ft", "linear_probe", "scratch"}
AUGMENTATIONS = {"light", "randaugment"}


@dataclass
class Config:
    # required
    name: str
    backbone: str
    pretrained: bool
    protocol: str
    augmentation: str
    lr: float
    epochs: int
    # defaults (spec §6)
    num_classes: int = 37
    img_size: int = 224
    batch_size: int = 32
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 2
    patience: int = 7
    seed: int = 42
    num_workers: int = 4
    val_fraction: float = 0.2
    data_root: str = "data"
    experiments_dir: str = "experiments"

    def validate(self) -> "Config":
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if self.augmentation not in AUGMENTATIONS:
            raise ValueError(f"augmentation must be one of {AUGMENTATIONS}, got {self.augmentation!r}")
        if self.protocol == "scratch" and self.pretrained:
            raise ValueError("protocol 'scratch' requires pretrained=False")
        if self.protocol in {"full_ft", "linear_probe"} and not self.pretrained:
            raise ValueError(f"protocol {self.protocol!r} requires pretrained=True")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(**data).validate()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: Config dataclass with YAML loader and validation"
```

---

## Task 5: The five variant config YAMLs (issue #2)

**Branch:** `feature/2-config-logging` (continues)

**Files:**
- Create: `configs/resnet50.yaml`, `configs/vit_b16_ft.yaml`, `configs/vit_s16_scratch.yaml`, `configs/deit_s16_ft.yaml`, `configs/vit_b16_linprobe.yaml`
- Test: extend `tests/test_config.py`

Backbone names are the exact strings the model factory (Task 7) dispatches on: `resnet50` (torchvision) and timm names `vit_base_patch16_224`, `vit_small_patch16_224`, `deit_small_patch16_224`.

- [ ] **Step 1: Write `configs/resnet50.yaml`** (variant #1 — CNN baseline)

```yaml
name: resnet50
backbone: resnet50
pretrained: true
protocol: full_ft
augmentation: light
lr: 0.0001
epochs: 20
```

- [ ] **Step 2: Write `configs/vit_b16_ft.yaml`** (variant #2 — ViT + transfer learning)

```yaml
name: vit_b16_ft
backbone: vit_base_patch16_224
pretrained: true
protocol: full_ft
augmentation: light
lr: 0.0001
epochs: 20
```

- [ ] **Step 3: Write `configs/vit_s16_scratch.yaml`** (variant #3 — from scratch)

`epochs: 60` because a from-scratch ViT needs more iterations than a fine-tuned one to converge at all; it still underperforms clearly, which is the intended narrative (spec §5). This is a hyperparameter choice, not a change to the variant matrix.

```yaml
name: vit_s16_scratch
backbone: vit_small_patch16_224
pretrained: false
protocol: scratch
augmentation: randaugment
lr: 0.0003
epochs: 60
patience: 12
```

- [ ] **Step 4: Write `configs/deit_s16_ft.yaml`** (variant #4 — data-efficient transformer)

```yaml
name: deit_s16_ft
backbone: deit_small_patch16_224
pretrained: true
protocol: full_ft
augmentation: light
lr: 0.0001
epochs: 20
```

- [ ] **Step 5: Write `configs/vit_b16_linprobe.yaml`** (variant #5 — linear probe)

`lr: 0.001` — a linear probe trains only the head, which tolerates (and benefits from) a higher LR than full fine-tuning. Spec §6 fixes LR only for fine-tuning and from-scratch, so this does not contradict it.

```yaml
name: vit_b16_linprobe
backbone: vit_base_patch16_224
pretrained: true
protocol: linear_probe
augmentation: light
lr: 0.001
epochs: 20
```

- [ ] **Step 6: Add a test that every shipped config loads and validates**

```python
# append to tests/test_config.py
import glob

def test_all_shipped_configs_are_valid():
    paths = sorted(glob.glob("configs/*.yaml"))
    assert len(paths) == 5
    names = {load_config(p).name for p in paths}
    assert names == {
        "resnet50", "vit_b16_ft", "vit_s16_scratch",
        "deit_s16_ft", "vit_b16_linprobe",
    }
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
git add configs/ tests/test_config.py
git commit -m "feat: five variant config YAMLs (spec §5)"
```

---

## Task 6: Data pipeline — split, transforms, loaders (issue #3)

**Branch:** `feature/3-data-pipeline`

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/2-config-logging -m "Merge feature/2-config-logging into develop"
git branch -d feature/2-config-logging
git checkout -b feature/3-data-pipeline develop
```

- [ ] **Step 2: Write the failing test for the pure split function**

```python
# tests/test_data.py
import pytest
import torch

from src.config import Config
from src.data import stratified_split, build_transforms


def _labels(n_classes=37, per_class=20):
    return [c for c in range(n_classes) for _ in range(per_class)]


def test_stratified_split_proportions_and_disjoint():
    labels = _labels()
    train_idx, val_idx = stratified_split(labels, val_fraction=0.2, seed=42)
    assert len(train_idx) + len(val_idx) == len(labels)
    assert set(train_idx).isdisjoint(val_idx)
    assert abs(len(val_idx) / len(labels) - 0.2) < 0.01


def test_stratified_split_is_deterministic():
    labels = _labels()
    a = stratified_split(labels, 0.2, seed=42)
    b = stratified_split(labels, 0.2, seed=42)
    assert a == b


def test_stratified_split_keeps_every_class_in_val():
    labels = _labels()
    _, val_idx = stratified_split(labels, 0.2, seed=42)
    val_classes = {labels[i] for i in val_idx}
    assert len(val_classes) == 37


def _cfg(**ov):
    base = dict(name="d", backbone="resnet50", pretrained=True,
                protocol="full_ft", augmentation="light", lr=1e-4, epochs=1)
    base.update(ov)
    return Config(**base)


def test_train_transform_outputs_correct_tensor_shape():
    from PIL import Image
    tf = build_transforms(_cfg(), train=True)
    out = tf(Image.new("RGB", (300, 250)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_eval_transform_outputs_correct_tensor_shape():
    from PIL import Image
    tf = build_transforms(_cfg(), train=False)
    out = tf(Image.new("RGB", (300, 250)))
    assert out.shape == (3, 224, 224)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 4: Implement `src/data.py`**

```python
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stratified_split(labels, val_fraction, seed):
    """Return (train_idx, val_idx) as lists, stratified by label."""
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    indices = list(range(len(labels)))
    train_idx, val_idx = next(splitter.split(indices, labels))
    return train_idx.tolist(), val_idx.tolist()


def build_transforms(cfg: Config, train: bool):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            normalize,
        ])
    ops = [
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.augmentation == "randaugment":
        ops.append(transforms.RandAugment())
    ops += [transforms.ToTensor(), normalize]
    return transforms.Compose(ops)


def get_dataloaders(cfg: Config):
    """Build train/val/test DataLoaders + class names for Oxford-IIIT Pets.

    Returns dict: {"train", "val", "test", "class_names"}.
    """
    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)

    # Two views of trainval so train/val get different transforms.
    trainval_train = OxfordIIITPet(
        cfg.data_root, split="trainval", target_types="category",
        transform=train_tf, download=True,
    )
    trainval_eval = OxfordIIITPet(
        cfg.data_root, split="trainval", target_types="category",
        transform=eval_tf, download=True,
    )
    test_ds = OxfordIIITPet(
        cfg.data_root, split="test", target_types="category",
        transform=eval_tf, download=True,
    )

    labels = list(trainval_train._labels)
    train_idx, val_idx = stratified_split(labels, cfg.val_fraction, cfg.seed)
    train_ds = Subset(trainval_train, train_idx)
    val_ds = Subset(trainval_eval, val_idx)

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=False, persistent_workers=cfg.num_workers > 0)
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
        "class_names": list(trainval_train.classes),
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: 5 passed (the `slow` dataset test below is deselected).

- [ ] **Step 6: Add a slow integration test for the real dataset**

```python
# append to tests/test_data.py
@pytest.mark.slow
def test_get_dataloaders_downloads_and_batches():
    loaders = get_dataloaders(_cfg(batch_size=8, num_workers=0))
    assert len(loaders["class_names"]) == 37
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 224, 224)
    assert targets.max().item() < 37
```

- [ ] **Step 7: Run the slow test once to confirm the download works**

Run: `pytest tests/test_data.py -m slow -v`
Expected: 1 passed (downloads ~800 MB into `data/` on first run).

- [ ] **Step 8: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: data pipeline — stratified split, transforms, dataloaders"
```

---

## Task 7: Model factory (issue #4)

**Branch:** `feature/4-model-factory`

**Files:**
- Create: `src/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/3-data-pipeline -m "Merge feature/3-data-pipeline into develop"
git branch -d feature/3-data-pipeline
git checkout -b feature/4-model-factory develop
```

- [ ] **Step 2: Write the failing test**

Tests use `pretrained=False` everywhere so they never download weights. Validation requires `pretrained=True` for non-scratch protocols, so the tests build `Config` objects and bypass `.validate()` deliberately.

```python
# tests/test_models.py
import torch

from src.config import Config
from src.models import build_model, get_gradcam_layer
from src.utils import count_parameters


def _cfg(backbone, protocol, **ov):
    base = dict(name="m", backbone=backbone, pretrained=False,
                protocol=protocol, augmentation="light", lr=1e-4, epochs=1)
    base.update(ov)
    return Config(**base)  # not validated: pretrained=False on purpose


def test_resnet_head_has_37_outputs():
    model = build_model(_cfg("resnet50", "full_ft"))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)


def test_vit_head_has_37_outputs():
    model = build_model(_cfg("vit_small_patch16_224", "scratch"))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)


def test_linear_probe_freezes_backbone_only():
    model = build_model(_cfg("vit_base_patch16_224", "linear_probe"))
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    assert 0 < trainable < total
    # trainable params are exactly the classifier head
    assert trainable == 37 * 768 + 37  # ViT-B hidden dim 768


def test_full_ft_keeps_all_params_trainable():
    model = build_model(_cfg("resnet50", "full_ft"))
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_get_gradcam_layer_returns_module():
    model = build_model(_cfg("resnet50", "full_ft"))
    layer = get_gradcam_layer(model)
    assert isinstance(layer, torch.nn.Module)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models'`.

- [ ] **Step 4: Implement `src/models.py`**

```python
import timm
import torch.nn as nn
import torchvision

from src.config import Config


def build_model(cfg: Config) -> nn.Module:
    """Build a backbone with a 37-class head behind one interface."""
    if cfg.backbone == "resnet50":
        weights = "IMAGENET1K_V2" if cfg.pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    else:  # timm ViT / DeiT
        model = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=cfg.num_classes
        )

    if cfg.protocol == "linear_probe":
        _freeze_backbone(model, cfg.backbone)
    return model


def _freeze_backbone(model: nn.Module, backbone: str) -> None:
    """Freeze every parameter, then re-enable the classifier head only."""
    for param in model.parameters():
        param.requires_grad = False
    head = model.fc if backbone == "resnet50" else model.get_classifier()
    for param in head.parameters():
        param.requires_grad = True


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    """Return the last conv block of a ResNet for Grad-CAM."""
    return model.layer4[-1]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: model factory over torchvision ResNet and timm ViT/DeiT"
```

---

## Task 8: Training loop (issue #5)

**Branch:** `feature/5-training-loop`

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/4-model-factory -m "Merge feature/4-model-factory into develop"
git branch -d feature/4-model-factory
git checkout -b feature/5-training-loop develop
```

- [ ] **Step 2: Write the failing smoke test (synthetic data, no download)**

```python
# tests/test_train.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import train_one_epoch, evaluate_model


def _tiny_loader(n=64, classes=4):
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _tiny_model(classes=4):
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, classes))


def test_train_one_epoch_reduces_loss():
    torch.manual_seed(0)
    model = _tiny_model()
    loader = _tiny_loader()
    device = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    first = train_one_epoch(model, loader, opt, crit, device, scheduler=None)
    last = first
    for _ in range(15):
        last = train_one_epoch(model, loader, opt, crit, device, scheduler=None)
    assert last["loss"] < first["loss"]


def test_evaluate_model_returns_loss_and_acc():
    model = _tiny_model()
    loader = _tiny_loader()
    crit = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(model, loader, crit, torch.device("cpu"))
    assert "loss" in metrics and "acc" in metrics
    assert 0.0 <= metrics["acc"] <= 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train'`.

- [ ] **Step 4: Implement `src/train.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: 2 passed.

- [ ] **Step 6: Smoke-test the CLI end to end on a tiny run**

Create `configs/_smoke.yaml` (a throwaway, not committed) with `epochs: 1`, then:
```bash
python -m src.train --config configs/_smoke.yaml
```
Expected: one epoch runs, `experiments/_smoke/` contains `best.pt`, `config.json`, `history.json`, `metadata.json`, `tb/`. Delete `configs/_smoke.yaml` and `experiments/_smoke/` afterward.

- [ ] **Step 7: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: training loop with cosine schedule, early stopping, TB logging"
```

---

## Task 9: Run the five variants (issues #6–#10)

**Branch:** `feature/6-10-experiment-runs`

This task produces no source code — it executes the pipeline and commits nothing but a short run log. Checkpoints/logs live under `experiments/` (gitignored). Each run trains on MPS; expect a long wall-clock time, longest for `vit_s16_scratch` (60 epochs from scratch).

- [ ] **Step 1: Create the branch**

```bash
git checkout develop
git merge --no-ff feature/5-training-loop -m "Merge feature/5-training-loop into develop"
git branch -d feature/5-training-loop
git checkout -b feature/6-10-experiment-runs develop
```

- [ ] **Step 2: Run variant #1 — `resnet50`**

Run: `python -m src.train --config configs/resnet50.yaml`
Expected: completes; `experiments/resnet50/best.pt` and `metadata.json` exist; `best_val_acc` is high (ResNet-50 fine-tune typically >0.90).

- [ ] **Step 3: Run variant #2 — `vit_b16_ft`**

Run: `python -m src.train --config configs/vit_b16_ft.yaml`
Expected: completes; `best_val_acc` high (ViT-B/16 fine-tune typically >0.92).

- [ ] **Step 4: Run variant #3 — `vit_s16_scratch`**

Run: `python -m src.train --config configs/vit_s16_scratch.yaml`
Expected: completes; `best_val_acc` clearly lower than fine-tuned variants (typically 0.30–0.55) — the intended data-hunger result.

- [ ] **Step 5: Run variant #4 — `deit_s16_ft`**

Run: `python -m src.train --config configs/deit_s16_ft.yaml`
Expected: completes; `best_val_acc` high.

- [ ] **Step 6: Run variant #5 — `vit_b16_linprobe`**

Run: `python -m src.train --config configs/vit_b16_linprobe.yaml`
Expected: completes; `best_val_acc` solid but typically below the full fine-tune of the same backbone.

- [ ] **Step 7: Record a run summary**

Create `experiments/RUNS.md` (gitignored content is fine, but this file is small — force-add it) summarizing each variant's `best_val_acc`, `epochs_run`, and `train_time_sec` read from the `metadata.json` files.

- [ ] **Step 8: Commit the run summary**

```bash
git add -f experiments/RUNS.md
git commit -m "chore: experiment run summary for the five variants"
```

> **Checkpoint:** if any run looks broken (e.g. fine-tuned variant stuck near chance, or an MPS op forcing a slow CPU fallback), stop and debug before continuing — downstream evaluation and figures depend on these checkpoints.

---

## Task 10: Evaluation & reporting (issue #11)

**Branch:** `feature/11-evaluation`

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/6-10-experiment-runs -m "Merge feature/6-10-experiment-runs into develop"
git branch -d feature/6-10-experiment-runs
git checkout -b feature/11-evaluation develop
```

- [ ] **Step 2: Write the failing test for the pure metric function**

```python
# tests/test_evaluate.py
from src.evaluate import compute_metrics


def test_compute_metrics_perfect_prediction():
    y_true = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_true)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]   # 3/4 correct
    m = compute_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 0.75) < 1e-9
    assert 0.0 <= m["macro_precision"] <= 1.0
    assert 0.0 <= m["macro_recall"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluate'`.

- [ ] **Step 4: Implement `src/evaluate.py`**

```python
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)

from src.config import Config
from src.data import get_dataloaders
from src.models import build_model
from src.train import evaluate_model
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: 2 passed.

- [ ] **Step 6: Evaluate all five finished runs**

Run, once per variant:
```bash
python -m src.evaluate --run-dir experiments/resnet50
python -m src.evaluate --run-dir experiments/vit_b16_ft
python -m src.evaluate --run-dir experiments/vit_s16_scratch
python -m src.evaluate --run-dir experiments/deit_s16_ft
python -m src.evaluate --run-dir experiments/vit_b16_linprobe
```
Expected: each run dir gains `test_metrics.json`, `confusion_matrix.png`, `curves.png`.

- [ ] **Step 7: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation — metrics, confusion matrix, training curves"
```

---

## Task 11: Grad-CAM for the CNN (issue #12)

**Branch:** `feature/12-gradcam`

**Files:**
- Create: `viz/gradcam.py`
- Test: `tests/test_gradcam.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/11-evaluation -m "Merge feature/11-evaluation into develop"
git branch -d feature/11-evaluation
git checkout -b feature/12-gradcam develop
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_gradcam.py
import torch
import torchvision

from viz.gradcam import GradCAM


def test_gradcam_heatmap_shape_and_range():
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model.eval()
    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam(torch.randn(1, 3, 224, 224))
    assert heatmap.shape == (224, 224)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0


def test_gradcam_respects_explicit_target_class():
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model.eval()
    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam(torch.randn(1, 3, 224, 224), target_class=5)
    assert heatmap.shape == (224, 224)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_gradcam.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz.gradcam'`.

- [ ] **Step 4: Implement `viz/gradcam.py`**

```python
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM over the last conv layer of a CNN (spec §8)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """image: (1,3,H,W). Returns a (H,W) heatmap normalized to [0,1]."""
        logits = self.model(image)
        if target_class is None:
            target_class = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, target_class].backward()

        # weight each channel by its mean gradient, then weighted-sum + ReLU
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_gradcam.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add viz/gradcam.py tests/test_gradcam.py
git commit -m "feat: Grad-CAM for the CNN"
```

---

## Task 12: Attention rollout for the ViT (issue #13)

**Branch:** `feature/13-attention-rollout`

**Files:**
- Create: `viz/attention_rollout.py`
- Test: `tests/test_attention_rollout.py`

timm's attention modules pass the post-softmax attention matrix into a dropout submodule named `attn_drop`. Disabling fused attention (`attn.fused_attn = False`) forces that explicit path, so a forward hook on each block's `attn.attn_drop` captures the attention weights.

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/12-gradcam -m "Merge feature/12-gradcam into develop"
git branch -d feature/12-gradcam
git checkout -b feature/13-attention-rollout develop
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_attention_rollout.py
import timm
import torch

from viz.attention_rollout import attention_rollout, ViTAttentionExtractor


def test_attention_rollout_pure_math_shape():
    # 2 layers, 1 head, 5 tokens (1 CLS + 4 patches)
    attns = [torch.rand(1, 1, 5, 5).softmax(-1) for _ in range(2)]
    heat = attention_rollout(attns, grid_size=2)
    assert heat.shape == (2, 2)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0


def test_extractor_collects_one_map_per_block():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=37)
    extractor = ViTAttentionExtractor(model)
    attns = extractor(torch.randn(1, 3, 224, 224))
    assert len(attns) == len(model.blocks)
    # ViT-S/16 @224: 196 patches + CLS = 197 tokens
    assert attns[0].shape[-1] == 197


def test_rollout_from_extractor_produces_14x14():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=37)
    extractor = ViTAttentionExtractor(model)
    attns = extractor(torch.randn(1, 3, 224, 224))
    heat = attention_rollout(attns, grid_size=14)
    assert heat.shape == (14, 14)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_attention_rollout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz.attention_rollout'`.

- [ ] **Step 4: Implement `viz/attention_rollout.py`**

```python
import torch
import torch.nn.functional as F


class ViTAttentionExtractor:
    """Captures post-softmax attention from every block of a timm ViT/DeiT."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self._attentions: list[torch.Tensor] = []
        for block in model.blocks:
            block.attn.fused_attn = False  # force explicit softmax path
            block.attn.attn_drop.register_forward_hook(self._save)

    def _save(self, _module, inp, _output):
        # inp[0] is the (B, heads, tokens, tokens) attention matrix
        self._attentions.append(inp[0].detach())

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> list[torch.Tensor]:
        self._attentions = []
        self.model(image)
        return list(self._attentions)


def attention_rollout(attentions: list[torch.Tensor], grid_size: int) -> torch.Tensor:
    """Roll attention across layers and return a (grid_size, grid_size) heatmap.

    attentions: list of (B, heads, tokens, tokens) post-softmax matrices.
    Adds the identity per layer for residual connections, re-normalizes,
    multiplies across layers, then reads the CLS->patches row.
    """
    rollout = None
    for attn in attentions:
        a = attn.mean(dim=1)[0]                 # average heads -> (tokens, tokens)
        a = a + torch.eye(a.size(0), device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout

    cls_to_patches = rollout[0, 1:]             # drop CLS->CLS
    heat = cls_to_patches.reshape(grid_size, grid_size)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat.cpu()


def rollout_heatmap_for_image(model, image: torch.Tensor, grid_size: int = 14) -> torch.Tensor:
    """Convenience: extract attention and roll it up for one image."""
    attentions = ViTAttentionExtractor(model)(image)
    return attention_rollout(attentions, grid_size)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_attention_rollout.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add viz/attention_rollout.py tests/test_attention_rollout.py
git commit -m "feat: attention rollout for the ViT"
```

---

## Task 13: Side-by-side figure generator (issue #14)

**Branch:** `feature/14-compare-figures`

**Files:**
- Create: `viz/compare_figures.py`
- Test: `tests/test_compare_figures.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/13-attention-rollout -m "Merge feature/13-attention-rollout into develop"
git branch -d feature/13-attention-rollout
git checkout -b feature/14-compare-figures develop
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_compare_figures.py
import torch

from viz.compare_figures import denormalize, overlay_heatmap


def test_denormalize_inverts_imagenet_normalization():
    from src.data import IMAGENET_MEAN, IMAGENET_STD
    from torchvision import transforms
    raw = torch.rand(3, 224, 224)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(raw)
    recovered = denormalize(norm)
    assert torch.allclose(recovered, raw, atol=1e-5)
    assert recovered.min() >= 0.0 and recovered.max() <= 1.0


def test_overlay_heatmap_returns_rgb_image():
    image = torch.rand(3, 224, 224)
    heat = torch.rand(224, 224)
    blended = overlay_heatmap(image, heat)
    assert blended.shape == (224, 224, 3)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_compare_figures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz.compare_figures'`.

- [ ] **Step 4: Implement `viz/compare_figures.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_compare_figures.py -v`
Expected: 2 passed.

- [ ] **Step 6: Generate the comparison figure from the real runs**

Run: `python -m viz.compare_figures --cnn-run experiments/resnet50 --vit-run experiments/vit_b16_ft`
Expected: `report/figures/comparison.png` exists and shows triplet rows.

- [ ] **Step 7: Commit**

```bash
git add viz/compare_figures.py tests/test_compare_figures.py report/figures/comparison.png
git commit -m "feat: side-by-side original/Grad-CAM/attention figure generator"
```

---

## Task 14: Aggregate final figures & results notebook (issue #15)

**Branch:** `feature/15-final-figures`

**Files:**
- Create: `notebooks/results_analysis.ipynb`
- Create (generated): `report/figures/results_table.csv`, `report/figures/accuracy_vs_params.png`, copied per-variant `confusion_matrix.png` / `curves.png`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/14-compare-figures -m "Merge feature/14-compare-figures into develop"
git branch -d feature/14-compare-figures
git checkout -b feature/15-final-figures develop
```

- [ ] **Step 2: Build the results notebook**

`notebooks/results_analysis.ipynb` — cells that:
1. Read every `experiments/<variant>/metadata.json` and `test_metrics.json`.
2. Assemble a results table: variant, total params, train time, best val acc, test acc, macro P/R/F1. Save to `report/figures/results_table.csv`.
3. Plot accuracy-vs-parameters (and accuracy-vs-train-time) scatter — the cost/accuracy trade-off (spec §7). Save `report/figures/accuracy_vs_params.png`.
4. Copy each variant's `confusion_matrix.png` and `curves.png` into `report/figures/<variant>_<name>.png`.
5. Display the `report/figures/comparison.png` triplet figure inline.

- [ ] **Step 3: Run the notebook end to end**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/results_analysis.ipynb`
Expected: completes with no errors; `report/figures/` is populated with the table CSV, the trade-off plot, and per-variant figures.

- [ ] **Step 4: Verify the aggregated figures exist**

Run: `ls report/figures/`
Expected: `comparison.png`, `results_table.csv`, `accuracy_vs_params.png`, and per-variant confusion-matrix/curve PNGs.

- [ ] **Step 5: Commit**

```bash
git add notebooks/results_analysis.ipynb report/figures/
git commit -m "feat: results notebook and aggregated report figures"
```

---

## Task 15: README, reproducibility polish & submission tag (issue #16)

**Branch:** `feature/16-readme-release`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout develop
git merge --no-ff feature/15-final-figures -m "Merge feature/15-final-figures into develop"
git branch -d feature/15-final-figures
git checkout -b feature/16-readme-release develop
```

- [ ] **Step 2: Write the full `README.md`**

Sections: project title + research question (1 paragraph); the five-variant table (from spec §5); Setup (venv + `pip install`); Reproduce (`python -m src.train --config configs/<variant>.yaml`, then `python -m src.evaluate`, then `python -m viz.compare_figures`); Results (embed `report/figures/comparison.png` and the results table); Repository layout; Tests (`pytest`); License.

- [ ] **Step 3: Run the full fast test suite**

Run: `pytest`
Expected: all non-slow tests pass.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: complete README with setup, reproduction and results"
```

- [ ] **Step 5: Merge to develop, then release into main**

```bash
git checkout develop
git merge --no-ff feature/16-readme-release -m "Merge feature/16-readme-release into develop"
git branch -d feature/16-readme-release
git checkout -b release/v1.0 develop
git checkout main
git merge --no-ff release/v1.0 -m "Release v1.0 — submission"
git tag -a v1.0-submission -m "University of Verona DL project — submission"
git checkout develop
git merge --no-ff release/v1.0 -m "Merge release/v1.0 back into develop"
git branch -d release/v1.0
```

- [ ] **Step 6: Final verification**

Run: `git tag` and `git log --oneline --graph --all | head -40`
Expected: `v1.0-submission` tag present on `main`; the Git Flow merge topology is visible.

---

## Self-Review

**Spec coverage** (`docs/project-description.md`):
- §3 dataset / official split + 80/20 stratified val → Task 6.
- §4 transfer learning, 224×224, ImageNet norm, light vs RandAugment, torchvision + timm, unified factory → Tasks 6, 7.
- §5 the five variants → Task 5 (configs) + Task 9 (runs).
- §6 hyperparameters → encoded in configs (Task 5) and the training loop defaults (Task 8).
- §7 evaluation: top-1, macro P/R/F1, confusion matrix, curves, params + runtime → Tasks 10, 14.
- §8 Grad-CAM, attention rollout, side-by-side figure → Tasks 11, 12, 13.
- §9 MPS device + CPU fallback, TensorBoard, seed/config logging → Tasks 2, 3, 8.
- §10 repository structure → Tasks 1–15 create every listed path.
- §11 Git Flow → branch/merge steps in every task.
- §12 milestones M1–M6 → Tasks map 1:1 to issues #1–#16.

**Open deviations from the spec** (intentional, flagged for review):
- `vit_s16_scratch` uses `epochs: 60` (spec §6 says "~20"); a from-scratch ViT needs more iterations to converge at all. Still underperforms fine-tuned variants — the intended narrative.
- `vit_b16_linprobe` uses `lr: 0.001` (spec §6 fixes LR only for fine-tuning/scratch); a linear probe trains only the head and tolerates a higher LR.
- A 6th variant (spec §5 "for margin") is **not** included — out of scope unless requested.

**Placeholder scan:** no TBD/TODO; every code step contains runnable code; the only prose-described artifact is the notebook (Task 14), which lists exact cell contents and output paths.

**Type consistency:** `Config` field names are consistent across `load_config`, `build_model`, `get_dataloaders`, `train`, `evaluate_variant`. `train_one_epoch`/`evaluate_model` return `{"loss", "acc"}` and are reused by `src.evaluate`. `attention_rollout(attentions, grid_size)` and `GradCAM(model, target_layer)` signatures match their call sites in `viz/compare_figures.py`.
