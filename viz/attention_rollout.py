import torch
import torch.nn.functional as F


class ViTAttentionExtractor:
    """Captures post-softmax attention from every block of a timm ViT/DeiT."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self._attentions: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for block in model.blocks:
            block.attn.fused_attn = False  # force explicit softmax path
            self._handles.append(
                block.attn.attn_drop.register_forward_hook(self._save))

    def _save(self, _module, inp, _output):
        self._attentions.append(inp[0].detach())

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> list[torch.Tensor]:
        self._attentions = []
        self.model(image)
        return list(self._attentions)

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


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
    """Convenience: extract attention, roll it up for one image, clean up hooks."""
    extractor = ViTAttentionExtractor(model)
    try:
        attentions = extractor(image)
        return attention_rollout(attentions, grid_size)
    finally:
        extractor.remove_hooks()
