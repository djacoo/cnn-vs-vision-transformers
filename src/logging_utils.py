import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    """Owns a single run directory: TensorBoard logs + JSON artifacts."""

    def __init__(self, experiments_dir: str, run_name: str):
        self.run_dir = Path(experiments_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def log_scalars(self, step: int, **scalars: float) -> None:
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)

    def log_figure(self, tag: str, figure, step: int = 0) -> None:
        self.writer.add_figure(tag, figure, step)

    def save_json(self, filename: str, payload: dict[str, Any]) -> None:
        (self.run_dir / filename).write_text(json.dumps(payload, indent=2))

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
