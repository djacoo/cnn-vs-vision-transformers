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
