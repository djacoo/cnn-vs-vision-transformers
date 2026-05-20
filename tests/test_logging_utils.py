import json

from src.logging_utils import RunLogger


def test_run_logger_creates_run_dir_and_writes(tmp_path):
    logger = RunLogger(experiments_dir=str(tmp_path), run_name="demo")
    assert logger.run_dir.exists()
    logger.log_scalars(step=0, train_loss=1.0, val_acc=0.5)
    logger.save_json("metadata.json", {"params": 42})
    logger.close()
    saved = json.loads((logger.run_dir / "metadata.json").read_text())
    assert saved["params"] == 42
    assert (logger.run_dir / "tb").exists()
