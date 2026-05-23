# References

Reference material consulted during development. PDFs of the core papers are mirrored alongside this file; the canonical sources are linked below.

## Papers

| Topic | Reference | Local PDF |
|---|---|---|
| Vision Transformer | Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, arXiv:2010.11929 | [vit-2010.11929.pdf](vit-2010.11929.pdf) |
| Data-Efficient Image Transformer | Touvron et al., *"Training data-efficient image transformers & distillation through attention"* (DeiT), arXiv:2012.12877 | [deit-2012.12877.pdf](deit-2012.12877.pdf) |
| Self-Supervised ViT | Caron et al., *"Emerging Properties in Self-Supervised Vision Transformers"* (DINO), arXiv:2104.14294 | [dino-2104.14294.pdf](dino-2104.14294.pdf) |
| Vision–Language | Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"* (CLIP), arXiv:2103.00020 | [clip-2103.00020.pdf](clip-2103.00020.pdf) |
| CNN saliency | Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, arXiv:1610.02391 | [gradcam-1610.02391.pdf](gradcam-1610.02391.pdf) |
| Attention visualization | Abnar & Zuidema, *"Quantifying Attention Flow in Transformers"* (attention rollout), arXiv:2005.00928 | [attention-rollout-2005.00928.pdf](attention-rollout-2005.00928.pdf) |

## Code repositories

| Library | Repository | Used for |
|---|---|---|
| `timm` | https://github.com/huggingface/pytorch-image-models | ViT-B/16, ViT-S/16, DeiT-S/16 backbones; DINO pretrained checkpoint (`vit_small_patch16_224.dino`) |
| `open_clip` | https://github.com/mlfoundations/open_clip | CLIP ViT-B/32 zero-shot inference (`ViT-B-32-quickgelu` OpenAI weights) |
| `torchvision` | https://github.com/pytorch/vision | ResNet-50 backbone, OxfordIIITPet dataset loader, standard transforms |
| DINO (original) | https://github.com/facebookresearch/dino | Reference for attention extraction methodology |

## Dataset

- **Oxford-IIIT Pet Dataset**: https://www.robots.ox.ac.uk/~vgg/data/pets/
- 37 cat/dog breed classes, ~7,400 images with per-image head bounding boxes (used in `src/pet_bboxes.py` for the quantitative saliency metrics).

## Library documentation

| Library | URL | Used for |
|---|---|---|
| PyTorch MPS backend | https://pytorch.org/docs/stable/notes/mps.html | Apple Silicon GPU training |
| timm model zoo | https://huggingface.co/timm | Pretrained checkpoints |
| scikit-learn `metrics` | https://scikit-learn.org/stable/modules/model_evaluation.html | Accuracy, macro precision/recall/F1, confusion matrix |
| scikit-learn `manifold.TSNE` | https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html | 2-D feature embedding |
| Matplotlib | https://matplotlib.org/ | All figures |

## Quantitative saliency metrics

- **Pointing Game**: Zhang et al., *"Top-down Neural Attention by Excitation Backprop"*, arXiv:1608.00507.
- **Deletion / Insertion AUC**: Petsiuk et al., *"RISE: Randomized Input Sampling for Explanation of Black-box Models"*, arXiv:1806.07421.
