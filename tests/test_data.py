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


@pytest.mark.slow
def test_get_dataloaders_downloads_and_batches():
    from src.data import get_dataloaders
    loaders = get_dataloaders(_cfg(batch_size=8, num_workers=0))
    assert len(loaders["class_names"]) == 37
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 224, 224)
    assert targets.max().item() < 37
