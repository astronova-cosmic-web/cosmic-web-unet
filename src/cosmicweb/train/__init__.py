"""Training: experiment configs, the training loop, checkpoints and logging.

    from cosmicweb.train import load_experiment, train

    cfg = load_experiment("configs/baseline.yaml", ["train.seed=1", "data.noise=low"])
    train(cfg)                     # or train(cfg, resume=True) after an interruption
"""

from .config import DEFAULTS, apply_overrides, load_experiment, run_name
from .trainer import METRIC_FIELDS, build_loss, load_model, train

__all__ = ["DEFAULTS", "METRIC_FIELDS", "apply_overrides", "build_loss", "load_experiment", "load_model",
           "run_name", "train"]
