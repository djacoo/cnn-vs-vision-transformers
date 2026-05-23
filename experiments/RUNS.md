# Experiment Run Summary

**Date range:** May 20–23, 2026
**Hardware:** MacBook Pro M3 Max, 48 GB unified memory, PyTorch MPS backend
**Seed:** 42 (deterministic dataloaders, fixed stratified val split)
**Dataset:** Oxford-IIIT Pets (37 classes, ~7,400 images; official trainval / test split; local 80/20 stratified train/val)

All checkpoints, TensorBoard logs, and per-run JSON artifacts live under `experiments/<variant>/` (gitignored except for this file and `.gitkeep`). The data-efficiency sweep lives under `experiments_data_eff/`. The implementation plan for the extensions is at `docs/superpowers/plans/2026-05-23-extensions-ssl-clip-saliency-data-efficiency.md`.

---

## 1. Baseline runs (5)

| # | Variant | Backbone | Protocol | Best val acc | Best epoch | Epochs run | Train time | Total params | Trainable | Test acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `resnet50` | ResNet-50 | full fine-tune | 0.9497 | 14 | 20 | 6m 17s | 23.58 M | 23.58 M | **0.9250** |
| 2 | `vit_b16_ft` | ViT-B/16 | full fine-tune | 0.9524 | 13 | 20 | 21m 26s | 85.83 M | 85.83 M | **0.9368** |
| 3 | `vit_s16_scratch` | ViT-S/16 | from scratch | 0.2826 | 48 | 60 | 19m 30s | 21.68 M | 21.68 M | **0.2270** |
| 4 | `deit_s16_ft` | DeiT-S/16 | full fine-tune | 0.9470 | 14 | 20 | 5m 45s | 21.68 M | 21.68 M | **0.9272** |
| 5 | `vit_b16_linprobe` | ViT-B/16 | linear probe | 0.9470 | 7 | 15 (early stop) | 5m 15s | 85.83 M | 0.028 M | **0.9242** |

Per-row notes:

1. **ResNet-50 — CNN reference.** Smooth convergence, early stopping triggered at epoch 14, no overfitting on the 5.9 k training images.
2. **ViT-B/16 fine-tuned — highest-capacity transformer baseline.** Highest test accuracy of all variants (0.937) but the longest train time at 21.4 min — diminishing returns vs DeiT-S at 1/4 the cost.
3. **ViT-S/16 from scratch — the data-hunger probe.** Runs to 60 epochs (its config doesn't early-stop on the same patience because val keeps drifting up slowly) and tops out at 0.283 val / 0.227 test. Without pretraining, a ViT cannot learn 37-way fine-grained classification from ~5.9 k images.
4. **DeiT-S/16 fine-tuned — Pareto-best architecture.** 0.927 test accuracy at 21.7 M params in 5.8 min. Architecturally identical to ViT-S but with stronger augmentation in the original DeiT recipe; reuses an ImageNet checkpoint trained with that recipe.
5. **ViT-B/16 linear probe — protocol study.** Only the 28 k-parameter head trains; backbone frozen. Early-stops at epoch 14, reaches 0.924 test in 5.2 min — within 0.3 pp of full fine-tune at 750× fewer trainable parameters. Frozen ImageNet features are already strongly transferable to fine-grained Pets.

**Cross-cutting interpretation.** All three pretrained, full-fine-tune variants (ResNet, ViT-B, DeiT-S) cluster within ~1 pp of each other on test accuracy — pretraining matters far more than the choice between CNN and transformer at this dataset scale. The 67-pp gap between #2 and #3 is the iconic ImageNet-pretraining effect.

---

## 2. Extension runs

| Variant / Analysis | Backbone | Protocol | Test acc | Macro F1 | Trainable | Train time | Notes |
|---|---|---|---|---|---|---|---|
| `vit_s16_dino_linprobe` | ViT-S/16 | DINO self-sup pretraining → linear probe | **0.9081** | 0.9076 | 0.014 M | 117 s | Best val acc 0.9375 at epoch 9; 17 epochs run before early stop |
| `clip_zeroshot` | CLIP ViT-B/32 | zero-shot inference (no training) | **0.8842** | — | 0 | inference only | 4-template prompt ensemble; CLIP's own preprocessing; `n_test=3669`; ViT-B-32-quickgelu weights to match OpenAI's QuickGELU activation (+4.94 pp vs non-QuickGELU checkpoint) |

CLIP templates (averaged in text-embedding space): `"a photo of a {}, a type of pet."`, `"a photo of a {}."`, `"a picture of a {} pet."`, `"an image of a {} cat or dog."`.

**Interpretation.** DINO self-supervised features (no labels in pretraining) match the supervised linear probe within 1.6 pp at half the parameters and 1/3 the training time — self-supervision transfers nearly losslessly on this dataset. CLIP, with *zero* fine-tuning, reaches 88.4% accuracy on 37 fine-grained breeds using the canonical QuickGELU weights; vision-language pretraining is a strong free baseline whenever class names are descriptive.

---

## 3. Saliency metrics runs (6)

200 validation images per variant; Grad-CAM for the CNN, attention rollout for ViT/DeiT/DINO. `vit_s16_scratch` is now included; see interpretation note below for its metric-artifact caveat.

| Variant | Pointing Game ↑ | Bbox-IoU @ top-20 ↑ | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---|---|---|---|
| `resnet50` | **0.780** | **0.340** | 0.246 | **0.760** |
| `vit_b16_ft` | 0.420 | 0.185 | 0.403 | 0.566 |
| `deit_s16_ft` | 0.605 | 0.267 | 0.351 | 0.648 |
| `vit_b16_linprobe` | 0.345 | 0.197 | 0.342 | 0.463 |
| `vit_s16_dino_linprobe` | 0.640 | 0.280 | **0.231** | 0.556 |
| `vit_s16_scratch` | 0.365 | 0.179 | 0.090 | 0.147 |

**Interpretation.** Grad-CAM on ResNet-50 substantially outperforms ViT attention rollout as a localizer (Pointing Game 0.78 vs 0.42 — a 36-point gap). Within ViTs, DINO self-supervised attention is the most object-centric (best deletion AUC across all variants), confirming the DINO paper's emergent-segmentation claim. **Attention ≠ saliency** is real and measurable.

`vit_s16_scratch` caveat: the deletion AUC of 0.090 is a measurement artifact. Deletion AUC is the area under the prob-vs-pixels-removed curve; when the model's baseline probability on the correct class is already near random (1/37 ≈ 0.027), that curve has almost no area mechanically — not because the saliency is good. The insertion AUC (0.147) is the honest metric: revealing the top-salient pixels barely recovers the prediction above chance, confirming there is no class-specific signal to recover.

---

## 4. Data-efficiency sweep (20 runs)

Five supervised variants × four training fractions (10 / 25 / 50 / 100%), stratified subsamples shared across variants per fraction. Test accuracy per cell.

| Variant | f = 0.10 | f = 0.25 | f = 0.50 | f = 1.00 |
|---|---|---|---|---|
| `resnet50` | 0.848 | 0.903 | 0.912 | 0.925 |
| `vit_b16_ft` | 0.864 | 0.914 | 0.929 | **0.937** |
| `vit_s16_scratch` | 0.065 | 0.113 | 0.135 | 0.227 |
| `deit_s16_ft` | 0.848 | 0.911 | 0.916 | 0.927 |
| `vit_b16_linprobe` | **0.886** | 0.909 | 0.925 | 0.924 |

Each cell is one full training run under the same recipe as the baseline (cosine LR + 2-epoch warmup, AdamW, label smoothing 0.1, early-stop on val acc with patience 5). The four fine-tune / linear-probe variants early-stop within their epoch budget at every fraction; `vit_s16_scratch` runs to its full epoch budget at every fraction without early stopping (val accuracy keeps creeping up, never plateaus high enough to trigger patience).

Raw per-run CSV: `experiments_data_eff/results.csv`.

**Interpretation.** ViT-B linear probe wins at 10% data (0.886) — frozen pretrained features have nothing to overfit. Full fine-tune ViT-B catches up and overtakes at 100% (0.937). From-scratch ViT never escapes its noise floor (0.227 even at full data). Curves: `report/figures/data_efficiency.png`.

---

## 5. Visualization runs (4)

| Figure | Source | Notes |
|---|---|---|
| `report/figures/dino_attention.png` | `viz/dino_attention.py` on `vit_s16_dino_linprobe` | 6 pet images × per-head + mean attention from last block |
| `report/figures/tsne_embeddings.png` | `viz/embeddings.py` | 4 variants (`resnet50`, `vit_b16_ft`, `vit_s16_scratch`, `vit_s16_dino_linprobe`), colored by class |
| `report/figures/failure_cases.png` | `viz/failures.py` | Top-5 most-confused breed pairs in ViT-B FT confusion matrix |
| `report/figures/saliency_metrics.png` | `viz/saliency_metrics_plot.py` | Bar chart of 4 metrics × 6 variants (scratch now included) |

---

## Reproducibility note

- Global seed `42` via `src/utils.set_seed`, deterministic dataloaders, fixed stratified train/val split.
- MPS backend on Apple Silicon; CPU fallback works elsewhere; CUDA path untested but should require no changes.
- `experiments/` is gitignored except for `RUNS.md` and `.gitkeep`. `experiments_data_eff/` holds the 20 data-efficiency run directories plus `results.csv` and is also kept out of version control.
