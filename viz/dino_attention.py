import torch

from viz.attention_rollout import ViTAttentionExtractor


@torch.no_grad()
def last_block_attention_heads(model, image: torch.Tensor,
                               grid_size: int = 14) -> torch.Tensor:
    """Per-head CLS->patches attention from the final transformer block.

    Returns (n_heads, grid_size, grid_size), each head normalized to [0, 1].
    """
    attentions = ViTAttentionExtractor(model)(image)
    return _heads_from_attentions(attentions, grid_size)


def _heads_from_attentions(attentions, grid_size):
    last = attentions[-1][0]                       # (heads, tokens, tokens)
    cls_to_patches = last[:, 0, 1:]                # (heads, num_patches)
    n_heads = cls_to_patches.size(0)
    maps = cls_to_patches.reshape(n_heads, grid_size, grid_size)
    mins = maps.amin(dim=(1, 2), keepdim=True)
    maxs = maps.amax(dim=(1, 2), keepdim=True)
    maps = (maps - mins) / (maxs - mins + 1e-8)
    return maps.cpu()


def render_dino_figure(model, images: torch.Tensor, out_path: str,
                       grid_size: int = 14) -> str:
    """Render a (n_images x (1 + n_heads + 1)) grid: original | per-head | mean.

    To generate report/figures/dino_attention.png after training:
        python -c "
    import torch, json
    from pathlib import Path
    from src.config import Config
    from src.models import build_model
    from src.data import get_dataloaders
    from src.utils import get_device, set_seed
    from viz.dino_attention import render_dino_figure
    run = Path('experiments/vit_s16_dino_linprobe')
    cfg = Config(**json.loads((run / 'config.json').read_text()))
    device = get_device()
    model = build_model(cfg).to(device)
    ckpt = torch.load(run / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    set_seed(cfg.seed)
    images, _ = next(iter(get_dataloaders(cfg)['test']))
    print(render_dino_figure(model, images[:6].to(device), 'report/figures/dino_attention.png'))
    "
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from viz.compare_figures import denormalize, overlay_heatmap

    rows = images.size(0)
    n_heads = model.blocks[-1].attn.num_heads
    cols = 1 + n_heads + 1

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if rows == 1:
        axes = axes[None, :]

    extractor = ViTAttentionExtractor(model)
    try:
        for r in range(rows):
            img = images[r:r + 1]
            heads = _heads_from_attentions(extractor(img), grid_size)
            axes[r, 0].imshow(denormalize(images[r]).permute(1, 2, 0))
            for h in range(n_heads):
                heat = torch.nn.functional.interpolate(
                    heads[h][None, None], size=images.shape[-2:],
                    mode="bilinear", align_corners=False)[0, 0]
                axes[r, 1 + h].imshow(overlay_heatmap(images[r], heat))
            mean_map = heads.mean(0)
            mean_map = torch.nn.functional.interpolate(
                mean_map[None, None], size=images.shape[-2:],
                mode="bilinear", align_corners=False)[0, 0]
            axes[r, -1].imshow(overlay_heatmap(images[r], mean_map))
            if r == 0:
                axes[r, 0].set_title("input")
                for h in range(n_heads):
                    axes[r, 1 + h].set_title(f"head {h}")
                axes[r, -1].set_title("mean")
            for c in range(cols):
                axes[r, c].axis("off")
    finally:
        extractor.remove_hooks()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
