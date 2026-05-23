import torch

from viz.embeddings import extract_features


def test_extract_features_returns_2d_tensor():
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 8 * 8, 16),
    )
    images = torch.randn(4, 3, 8, 8)
    feats = extract_features(model, images, layer=None, device=torch.device("cpu"))
    assert feats.shape == (4, 16)
