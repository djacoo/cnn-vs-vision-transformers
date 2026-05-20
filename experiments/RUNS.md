# Experiment Run Summary

Five variants trained on Oxford-IIIT Pets (37 classes), MacBook Pro M3 Max, PyTorch MPS backend.
Metrics below are **validation** accuracy (held-out test metrics are produced by `src/evaluate.py`).

| # | Variant | Backbone | Protocol | Best val acc | Best epoch | Epochs run | Train time | Total params | Trainable params |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `resnet50` | ResNet-50 | full fine-tune | 0.9497 | 14 | 20 | 6m 17s | 23.58M | 23.58M |
| 2 | `vit_b16_ft` | ViT-B/16 | full fine-tune | 0.9524 | 13 | 20 | 21m 26s | 85.83M | 85.83M |
| 3 | `vit_s16_scratch` | ViT-S/16 | from scratch | 0.2826 | 48 | 60 | 19m 30s | 21.68M | 21.68M |
| 4 | `deit_s16_ft` | DeiT-S/16 | full fine-tune | 0.9470 | 14 | 20 | 5m 45s | 21.68M | 21.68M |
| 5 | `vit_b16_linprobe` | ViT-B/16 | linear probe | 0.9470 | 7 | 15 (early stop) | 5m 15s | 85.83M | 0.028M |

## Notes

- **CNN vs transformer under transfer learning (1 vs 2/4):** all three fine-tuned variants land within ~0.5 pp of each other (0.947–0.952). Comparable accuracy, different mechanisms.
- **Effect of pretraining (2 vs 3):** the from-scratch ViT-S reaches only 0.283 vs 0.952 for the fine-tuned ViT-B — a 67 pp gap. This is the data-hunger result: a ViT without inductive biases cannot learn a fine-grained task from ~5.9k training images alone.
- **Data efficiency (2 vs 4):** DeiT-S (21.7M params) matches ViT-B (85.8M) within 0.5 pp at ~4× fewer params and ~4× less training time.
- **Full fine-tune vs linear probe (2 vs 5):** linear probing only the 28k-parameter head reaches 0.947 — within 0.5 pp of full fine-tuning — and early-stopped at epoch 14. Frozen ImageNet features are already strongly transferable to Pets.

All checkpoints, TensorBoard logs, and per-run JSON artifacts live under `experiments/<variant>/` (gitignored).
