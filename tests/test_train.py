import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import train_one_epoch, evaluate_model


def _tiny_loader(n=64, classes=4):
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _tiny_model(classes=4):
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, classes))


def test_train_one_epoch_reduces_loss():
    torch.manual_seed(0)
    model = _tiny_model()
    loader = _tiny_loader()
    device = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    first = train_one_epoch(model, loader, opt, crit, device, scheduler=None)
    last = first
    for _ in range(15):
        last = train_one_epoch(model, loader, opt, crit, device, scheduler=None)
    assert last["loss"] < first["loss"]


def test_evaluate_model_returns_loss_and_acc():
    model = _tiny_model()
    loader = _tiny_loader()
    crit = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(model, loader, crit, torch.device("cpu"))
    assert "loss" in metrics and "acc" in metrics
    assert 0.0 <= metrics["acc"] <= 1.0
