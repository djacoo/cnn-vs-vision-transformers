from dataclasses import asdict, dataclass

import yaml

PROTOCOLS = {"full_ft", "linear_probe", "scratch"}
AUGMENTATIONS = {"light", "randaugment"}


@dataclass
class Config:
    # required
    name: str
    backbone: str
    pretrained: bool
    protocol: str
    augmentation: str
    lr: float
    epochs: int
    # defaults (spec §6)
    num_classes: int = 37
    img_size: int = 224
    batch_size: int = 32
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 2
    patience: int = 7
    seed: int = 42
    num_workers: int = 4
    val_fraction: float = 0.2
    train_fraction: float = 1.0
    data_root: str = "data"
    experiments_dir: str = "experiments"

    def validate(self) -> "Config":
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if self.augmentation not in AUGMENTATIONS:
            raise ValueError(f"augmentation must be one of {AUGMENTATIONS}, got {self.augmentation!r}")
        if self.protocol == "scratch" and self.pretrained:
            raise ValueError("protocol 'scratch' requires pretrained=False")
        if self.protocol in {"full_ft", "linear_probe"} and not self.pretrained:
            raise ValueError(f"protocol {self.protocol!r} requires pretrained=True")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1]")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(**data).validate()
