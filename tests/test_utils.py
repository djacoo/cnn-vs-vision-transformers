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
