# CNN vs Vision Transformer — Where Does the Model Look?

> **Deep Learning project — University of Verona, A.Y. 2025–26**
> A comparative study of CNN and Vision Transformer image classifiers on a fine-grained
> dataset, focused not only on *which* model is more accurate but on *where* each model
> looks when it decides — visualized with Grad-CAM and attention rollout.

This document is the single source of truth for the project. It is written to double as a
context file for Claude Code (you can rename or symlink it to `CLAUDE.md` at the repo root).

---

## 1. Overview & research question

CNNs and Vision Transformers (ViTs) reach comparable accuracy on image classification but get
there through fundamentally different mechanisms:

- A **CNN** has strong built-in inductive biases — locality and translation equivariance —
  so it learns efficiently from small datasets.
- A **ViT** has almost none of that. It splits the image into patches, flattens them, adds
  positional embeddings, and feeds the sequence to a standard transformer encoder, learning
  spatial relationships from scratch via self-attention. The known consequence is that ViTs
  are pretrained on large labeled datasets and then fine-tuned downstream; without that
  pretraining they tend to underperform on small data.

**The project demonstrates this trade-off empirically and then visualizes how the two
architectures attend to the same image.** The visualizations are the centerpiece of the oral
presentation.

---

## 2. Exam requirements this project satisfies

| Requirement | How this project meets it |
|---|---|
| Deep models from the lectures | CNNs, ViT, multi-head self-attention, transfer learning |
| **≥ 5 compared variants** | 5 model/protocol variants (see §5) — the comparison *is* the experiment matrix |
| Dataset constrained to the task | Oxford-IIIT Pets (fine-grained classification) |
| Code shared via GitHub | Git Flow + issue-based repo (see §9–§11) |
| Written technical report (8–20 pp) | Maps directly onto the required report sections (see §13) |
| Oral presentation (8–10 min) + slides | Side-by-side saliency figures as the highlight |
| Theory Q&A | Sets up attention/transformer/transfer-learning questions (see §14) |

---

## 3. Dataset — Oxford-IIIT Pets

- **37 classes** (cat & dog breeds), **~7,400 images**, roughly balanced.
- Available directly via `torchvision.datasets.OxfordIIITPet`.
- Chosen because each image has a **single clear foreground animal**, so Grad-CAM and
  attention maps localize cleanly and read instantly on a slide.
- Split: use the official trainval/test split; carve a validation set out of trainval
  (e.g. 80/20 stratified by class). Fix the seed for reproducibility.

---

## 4. Methodology

- **Transfer learning** is the backbone of the study: most variants start from
  ImageNet-pretrained weights and are fine-tuned on Pets.
- Unified input pipeline: resize/crop to **224×224**, normalize with ImageNet statistics,
  light augmentation for fine-tuned models and stronger augmentation (RandAugment) for the
  from-scratch ViT.
- Models sourced from `torchvision` (ResNet) and `timm` (ViT / DeiT).
- A single `model factory` exposes every backbone behind one interface so the training loop
  is identical across variants.

---

## 5. The five variants (hard requirement — pinned)

| # | Config name | Backbone | Pretrained | Protocol | Role in the story |
|---|---|---|---|---|---|
| 1 | `resnet50` | ResNet-50 | ImageNet | full fine-tune | CNN baseline |
| 2 | `vit_b16_ft` | ViT-B/16 | ImageNet-21k→1k | full fine-tune | Transformer + transfer learning |
| 3 | `vit_s16_scratch` | ViT-S/16 | none | from scratch | Isolates pretraining → shows data-hunger |
| 4 | `deit_s16_ft` | DeiT-S/16 | ImageNet | full fine-tune | Data-efficient / smaller transformer |
| 5 | `vit_b16_linprobe` | ViT-B/16 | ImageNet | frozen backbone, head only | Protocol study vs #2 |

**Comparison axes (what each contrast proves):**
- **1 vs 2 / 4** — CNN vs transformer under transfer learning.
- **2 vs 3** — isolates the effect of pretraining (the data-hunger point). A weak result for
  #3 is a *good* result for the narrative.
- **2 vs 4** — isolates model size / data efficiency.
- **2 vs 5** — full fine-tune vs linear probe (controlled protocol study).

A 6th variant (e.g. ViT patch size 16 vs 32) can be added for margin.

---

## 6. Hyperparameters (shared starting point — to be tuned)

- Input 224×224, batch size 32.
- Optimizer: AdamW, weight decay 0.05.
- LR ≈ 1e-4 for fine-tuning; ≈ 3e-4 for the from-scratch ViT.
- Cosine LR schedule with short warmup.
- Label smoothing 0.1.
- ~20 epochs with early stopping on validation accuracy.
- Stronger augmentation (RandAugment) only for `vit_s16_scratch`.

---

## 7. Evaluation

- **Accuracy** (top-1), **macro precision/recall/F1**.
- **Confusion matrix** per variant (37×37).
- **Training/validation curves** (loss & accuracy).
- **Parameter count** and **training time per model** → discuss the accuracy-vs-cost
  trade-off.
- Report all metrics on the held-out test split only, once, after model selection on
  validation.

---

## 8. Visualizations — the centerpiece

Two complementary techniques, shown side by side over the same images.

**Grad-CAM (CNN).** Uses the gradient of the predicted class flowing into the last
convolutional layer to produce a coarse heatmap over the image — "which region drove this
prediction." Simple and legible.

**Attention rollout (ViT).** The principled way to visualize a transformer. Self-attention
computes, per layer, how much each patch token attends to every other (the QKᵀ/√dₖ score
matrix, softmax-normalized). Rollout multiplies these attention matrices across layers
(adding the identity to account for residual connections, then re-normalizing) to trace how
much each image patch ultimately contributes to the classification (CLS) token, yielding a
heatmap over patches. Per-head/per-layer maps can also be shown to illustrate that different
heads attend to different regions.

**The money shot:** rows of triplets — `original image | CNN Grad-CAM | ViT attention map`.
This makes the inductive-bias difference visible at a glance (the CNN often fixates on a
discriminative texture patch; the ViT spreads attention more globally).

---

## 9. Hardware, environment & stack

- **Machine:** MacBook Pro M3 Max, 48 GB unified memory.
- **Compute:** PyTorch **MPS** (Metal) backend. Always select `mps` if available, else CPU.
  All variants train comfortably at this scale; watch for the occasional op falling back to
  CPU and swap the model/op if a run is unexpectedly slow.
- **Core libraries:** `torch`, `torchvision`, `timm`, `numpy`, `scikit-learn`
  (metrics/confusion matrix), `matplotlib`, `pyyaml`.
- **Experiment tracking:** **TensorBoard** (local, no account) — log scalars (loss/acc/LR),
  the confusion matrix, and sample saliency figures as images.
- **Reproducibility:** global seed utility; log the resolved config with every run.

---

## 10. Repository structure

```
pets-cnn-vs-vit/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── PROJECT.md                # this file (project spec / Claude Code context)
├── configs/                  # one YAML per variant (§5)
│   ├── resnet50.yaml
│   ├── vit_b16_ft.yaml
│   ├── vit_s16_scratch.yaml
│   ├── deit_s16_ft.yaml
│   └── vit_b16_linprobe.yaml
├── src/
│   ├── config.py             # load/validate YAML configs
│   ├── data.py               # Pets download, splits, transforms, DataLoaders
│   ├── models.py             # model factory over torchvision + timm
│   ├── train.py              # training loop, checkpointing, early stopping
│   ├── evaluate.py           # metrics, confusion matrix, curves
│   └── utils.py              # device (MPS), seeding, logging
├── viz/
│   ├── gradcam.py            # Grad-CAM for the CNN
│   ├── attention_rollout.py  # attention rollout + per-head maps for the ViT
│   └── compare_figures.py    # original | Grad-CAM | attention figure generator
├── notebooks/
│   └── results_analysis.ipynb
├── experiments/              # gitignored: checkpoints, TensorBoard logs, figures
└── report/
    └── figures/              # final figures exported for the report/slides
```

---

## 11. Git Flow workflow

Branching model:
- `main` — protected, always stable; receives merges only via release branches. Tag the
  final submission state `v1.0-submission`.
- `develop` — integration branch; default working branch.
- `feature/<issue#>-<short-name>` — one branch per issue, off `develop`, merged back via PR,
  then deleted.
- `release/v1.0` — cut from `develop` when experiments are done, for the report/figure
  freeze; merged into both `main` and `develop`.

Per-issue loop:
```bash
git checkout develop && git pull
git checkout -b feature/3-data-pipeline develop
# ... work, commit ...
git push -u origin feature/3-data-pipeline
# open a PR into develop, review, merge, delete branch, close issue
```

---

## 12. Milestones & issues

**M1 — Infrastructure**
- #1 Repo init: README, `.gitignore`, `requirements.txt`, MPS device check, seed utility
- #2 Config system (YAML per variant) + TensorBoard logging
- #3 Data pipeline: download Oxford-IIIT Pets, stratified train/val/test split, transforms, DataLoaders

**M2 — Models & training**
- #4 Model factory (unified interface over ResNet / ViT / DeiT)
- #5 Training loop: train/eval, checkpointing, early stopping, metric logging (MPS-aware)

**M3 — Experiments (the 5 variants)**
- #6 Run `resnet50`
- #7 Run `vit_b16_ft`
- #8 Run `vit_s16_scratch`
- #9 Run `deit_s16_ft`
- #10 Run `vit_b16_linprobe`

**M4 — Evaluation**
- #11 Metrics & reporting: accuracy, precision/recall/F1, confusion matrices, curves, params + runtime table

**M5 — Visualizations (centerpiece)**
- #12 Grad-CAM for the CNN
- #13 Attention rollout + per-head maps for the ViT
- #14 Side-by-side figure generator (original | Grad-CAM | attention)

**M6 — Deliverables**
- #15 Aggregate final figures for report/slides
- #16 README + reproducibility polish, tag `v1.0-submission`

---

## 13. Mapping to the required report sections

- **Motivation & Rationale** — the CNN-vs-transformer mechanism question; why interpretability matters.
- **State of the Art** — ViT, DeiT, transfer learning, Grad-CAM, attention rollout.
- **Objectives** — quantify the accuracy trade-off and characterize *how* each model attends.
- **Methodology** — transfer learning + the 5-variant matrix + visualization techniques.
- **Experiments & Results** — ablation table, curves, confusion matrices, saliency figures.
- **Conclusions** — findings + future work (more datasets, other ViT variants, quantitative attention metrics).
- **References** — only cited works.

---

## 14. Theory Q&A — topics to prepare

- Self-attention: the QKᵀ/√dₖ formula, why the scaling factor, softmax → attention weights.
- Multi-head attention and why multiple heads help.
- ViT specifics: patch embedding, CLS token, positional embeddings, transformer encoder.
- Why a ViT lacks the CNN's inductive biases (locality, translation equivariance) and what that implies for data efficiency.
- Transfer learning vs from-scratch training; full fine-tune vs linear probing.
- How Grad-CAM and attention rollout work and what each actually measures.

---

## 15. Constraints & boundaries

- **The report must be written by you.** The course checks reports with AI-text detectors;
  AI tools may only be used to fix typos / rephrase (Grammarly is recommended as it is not
  flagged). This spec and the code are project assets, not report prose — do not paste
  generated prose into the report.
- **Repo creation, authentication, and pushing are yours to do.** Account creation and
  credential-based actions must be performed by you; Claude/Claude Code will write code and
  provide exact `git`/`gh` commands but will not authenticate or create accounts on your
  behalf.
- Keep `experiments/` out of version control (large checkpoints/logs); commit only configs,
  code, and the final exported figures under `report/figures/`.

---

## 16. Working conventions for Claude Code

- Follow **Git Flow**: one feature branch per issue, off `develop`, PR back into `develop`.
- Work **issue by issue** in milestone order (M1 → M6); don't start a variant run before the
  training loop (#5) is merged.
- Ask before introducing new dependencies or changing the variant matrix in §5.
- Prefer the `mps` device; guard every device-specific path with a CPU fallback.
- Log every run to TensorBoard and save the resolved config alongside checkpoints.
