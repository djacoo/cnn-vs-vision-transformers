import pytest

from src.config import Config, load_config


def _valid_kwargs(**overrides):
    base = dict(
        name="demo", backbone="resnet50", pretrained=True,
        protocol="full_ft", augmentation="light", lr=1e-4, epochs=20,
    )
    base.update(overrides)
    return base


def test_config_defaults():
    cfg = Config(**_valid_kwargs())
    assert cfg.num_classes == 37
    assert cfg.img_size == 224
    assert cfg.batch_size == 32
    assert cfg.weight_decay == 0.05
    assert cfg.label_smoothing == 0.1
    assert cfg.seed == 42


def test_config_rejects_unknown_protocol():
    with pytest.raises(ValueError):
        Config(**_valid_kwargs(protocol="bogus")).validate()


def test_config_rejects_pretrained_scratch_mismatch():
    # scratch protocol must not use pretrained weights
    with pytest.raises(ValueError):
        Config(**_valid_kwargs(protocol="scratch", pretrained=True)).validate()
    # fine-tune / linear probe require pretrained weights
    with pytest.raises(ValueError):
        Config(**_valid_kwargs(protocol="full_ft", pretrained=False)).validate()


def test_load_config_roundtrip(tmp_path):
    yaml_text = (
        "name: t\nbackbone: resnet50\npretrained: true\n"
        "protocol: full_ft\naugmentation: light\nlr: 0.0001\nepochs: 3\n"
    )
    path = tmp_path / "t.yaml"
    path.write_text(yaml_text)
    cfg = load_config(str(path))
    assert cfg.name == "t" and cfg.epochs == 3 and cfg.protocol == "full_ft"


import glob

def test_all_shipped_configs_are_valid():
    paths = sorted(glob.glob("configs/*.yaml"))
    assert len(paths) == 5
    names = {load_config(p).name for p in paths}
    assert names == {
        "resnet50", "vit_b16_ft", "vit_s16_scratch",
        "deit_s16_ft", "vit_b16_linprobe",
    }
