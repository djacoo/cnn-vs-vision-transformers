from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stratified_split(labels, val_fraction, seed):
    """Return (train_idx, val_idx) as lists, stratified by label."""
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    indices = list(range(len(labels)))
    train_idx, val_idx = next(splitter.split(indices, labels))
    return train_idx.tolist(), val_idx.tolist()


def build_transforms(cfg: Config, train: bool):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            normalize,
        ])
    ops = [
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.augmentation == "randaugment":
        ops.append(transforms.RandAugment())
    ops += [transforms.ToTensor(), normalize]
    return transforms.Compose(ops)


def get_dataloaders(cfg: Config):
    """Build train/val/test DataLoaders + class names for Oxford-IIIT Pets.

    Returns dict: {"train", "val", "test", "class_names"}.
    """
    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)

    # Two views of trainval so train/val get different transforms.
    trainval_train = OxfordIIITPet(
        cfg.data_root, split="trainval", target_types="category",
        transform=train_tf, download=True,
    )
    trainval_eval = OxfordIIITPet(
        cfg.data_root, split="trainval", target_types="category",
        transform=eval_tf, download=True,
    )
    test_ds = OxfordIIITPet(
        cfg.data_root, split="test", target_types="category",
        transform=eval_tf, download=True,
    )

    labels = list(trainval_train._labels)
    train_idx, val_idx = stratified_split(labels, cfg.val_fraction, cfg.seed)
    train_ds = Subset(trainval_train, train_idx)
    val_ds = Subset(trainval_eval, val_idx)

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=False, persistent_workers=cfg.num_workers > 0)
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
        "class_names": list(trainval_train.classes),
    }
