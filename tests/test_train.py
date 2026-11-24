import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from cosmicweb.train import load_experiment, load_model, run_name, train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TIMING = {"epoch_time_s", "train_items_per_s"}


def experiment(tmp_path, built, overrides=(), name="exp", **sections):
    """A tiny CPU experiment on the synthetic 64x64 dataset from conftest."""
    data_cfg, _ = built
    raw = {
        "data": {"config": str(data_cfg.path), "noise": "high", "mask": "mid",
                 "train": {"mode": "random", "patch_size": 32, "crops_per_map": 2, "augment": True,
                           "resample_each_epoch": True},
                 "val": {"mode": "tiles", "patch_size": 32}, "batch_size": 4, "num_workers": 0},
        "model": {"base_channels": 4, "depth": 2},
        "train": {"epochs": 3, "seed": 0},
        "output": {"runs_dir": str(tmp_path / "runs"), "tensorboard": False},
    }
    for section, values in sections.items():
        raw[section].update(values)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_experiment(path, list(overrides))


def run_dir(cfg):
    return Path(cfg["output"]["runs_dir"]) / run_name(cfg)


def metrics(cfg, drop=TIMING):
    with (run_dir(cfg) / "metrics.csv").open(newline="") as fh:
        return [{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(fh)]


def weights(cfg, which="last"):
    return torch.load(run_dir(cfg) / "checkpoints" / f"{which}.pt", weights_only=True)["model"]


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- config -------------------------------------------------------------------------------------

def test_config_defaults_overrides_and_paths(tmp_path, built):
    cfg = experiment(tmp_path, built, ["train.seed=3", "optim.lr=1e-4", "data.noise=low", "train.grad_clip=1"])
    assert cfg["train"]["seed"] == 3 and cfg["data"]["noise"] == "low"
    assert cfg["optim"]["lr"] == 1e-4 and isinstance(cfg["optim"]["lr"], float)  # "1e-4" is a string in YAML
    assert cfg["model"]["norm"] == "batch"  # default filled in
    assert cfg["experiment"] == "exp" and run_name(cfg) == "exp_low-mid_seed3"
    assert Path(cfg["data"]["config"]).is_absolute()


@pytest.mark.parametrize("bad", [["train.sed=1"], ["nosuch.key=1"], ["train.seed"], ["optim.lr=fast"],
                                 ["train.early_stopping_patience=0"], ["optim.lr=1e-6"]])  # last: below min_lr
def test_bad_overrides(tmp_path, built, bad):
    with pytest.raises(ValueError):
        experiment(tmp_path, built, bad)


def test_unknown_key_in_file(tmp_path, built):
    with pytest.raises(ValueError, match="epoch"):
        experiment(tmp_path, built, train={"epoch": 3})


def test_baseline_config_loads():
    cfg = load_experiment(REPO / "configs" / "baseline.yaml")
    assert Path(cfg["data"]["config"]) == (REPO / "configs" / "data.yaml").resolve()
    assert run_name(cfg) == "baseline_high-mid_seed0"


# --- training -----------------------------------------------------------------------------------

def test_training_run_writes_everything(tmp_path, built):
    cfg = experiment(tmp_path, built, ["output.tensorboard=true"])
    info = train(cfg, verbose=False)
    d = run_dir(cfg)
    assert {"config.yaml", "run.json", "metrics.csv", "checkpoints", "tensorboard"} <= {p.name for p in d.iterdir()}
    assert (d / "checkpoints" / "last.pt").exists() and (d / "checkpoints" / "best.pt").exists()
    assert any((d / "tensorboard").iterdir())
    rows = metrics(cfg, drop=set())
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]
    assert all(float(r["train_loss"]) > 0 and float(r["val_loss"]) > 0 for r in rows)
    assert info["epochs_completed"] == 3 and info["stop_reason"] == "completed"
    assert info["dataset_fingerprint"] == built[1]["fingerprint"]
    assert info["best_epoch"] in (1, 2, 3) and info["checkpoint_mb"] > 0
    assert info["train_items_per_epoch"] == 12 and info["val_items"] == 8
    assert json.loads((d / "run.json").read_text()) == info
    assert yaml.safe_load((d / "config.yaml").read_text()) == cfg

    model = load_model(d / "checkpoints" / "best.pt")
    assert not model.training
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_same_seed_reproduces_and_other_seed_differs(tmp_path, built):
    a = experiment(tmp_path, built, ["output.name=a"])
    b = experiment(tmp_path, built, ["output.name=b"])
    c = experiment(tmp_path, built, ["output.name=c", "train.seed=1"])
    for cfg in (a, b, c):
        train(cfg, verbose=False)
    assert metrics(a) == metrics(b) and same_weights(weights(a), weights(b))
    assert not same_weights(weights(a), weights(c))


def test_resume_after_interruption_matches_uninterrupted_run(tmp_path, built):
    full = experiment(tmp_path, built, ["output.name=full"])
    train(full, verbose=False)

    split = experiment(tmp_path, built, ["output.name=split"])
    first = train(split, max_epochs_this_session=1, verbose=False)
    assert first["stop_reason"] == "session limit" and first["epochs_completed"] == 1
    with (run_dir(split) / "metrics.csv").open("a", newline="") as fh:  # epoch logged, but crash before its checkpoint
        fh.write("2,9,9,9,9,9,9,\n")
    info = train(split, resume=True, verbose=False)

    assert metrics(split) == metrics(full)  # the orphan row was replaced, not duplicated
    assert same_weights(weights(split), weights(full))
    assert same_weights(weights(split, "best"), weights(full, "best"))
    assert info["epochs_completed"] == 3 and len(info["sessions"]) == 2


def test_run_folder_guards(tmp_path, built):
    cfg = experiment(tmp_path, built, ["train.epochs=1"])
    with pytest.raises(FileNotFoundError):
        train(cfg, resume=True, verbose=False)
    train(cfg, verbose=False)
    with pytest.raises(FileExistsError):
        train(cfg, verbose=False)
    assert train(cfg, resume="auto", verbose=False)["epochs_completed"] == 1  # nothing left to do

    changed = experiment(tmp_path, built, ["train.epochs=1", "output.name=" + run_name(cfg), "data.mask=none"])
    with pytest.raises(ValueError, match="data.mask"):
        train(changed, resume=True, verbose=False)
    ok = experiment(tmp_path, built, ["train.epochs=1", "output.name=" + run_name(cfg), "data.num_workers=1"])
    train(ok, resume=True, verbose=False)  # machine-specific settings may change


def test_early_stopping(tmp_path, built):
    # lr 0 and no BatchNorm (its running statistics would still change): the model is frozen, so the
    # validation loss never improves after epoch 1
    cfg = experiment(tmp_path, built, ["optim.lr=0.0", "schedule.name=constant", "model.norm=none",
                                       "train.early_stopping_patience=1", "train.epochs=5"])
    info = train(cfg, verbose=False)
    assert info["stop_reason"] == "early stopping" and info["epochs_completed"] == 2 and info["best_epoch"] == 1


def test_divergence_is_reported(tmp_path, built):
    cfg = experiment(tmp_path, built, ["optim.lr=1e30", "optim.name=adam"])
    with pytest.raises(FloatingPointError, match="grad_clip"):
        train(cfg, verbose=False)


def test_validation_must_be_fixed(tmp_path, built):
    cfg = experiment(tmp_path, built, ["data.val.augment=true"])
    with pytest.raises(ValueError, match="validation"):
        train(cfg, verbose=False)


def test_unknown_loss(tmp_path, built):
    cfg = experiment(tmp_path, built, ["loss.name=spectrum"])
    with pytest.raises(ValueError, match="unknown loss"):
        train(cfg, verbose=False)


def test_command_line(tmp_path, built):
    cfg = experiment(tmp_path, built, ["train.epochs=1"])
    exp = tmp_path / "exp.yaml"
    runs = tmp_path / "cli runs"  # a path with a space
    cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(exp), "--runs-dir", str(runs),
           "--set", "train.epochs=1", "--set", "train.seed=4"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "epoch   1/1" in out.stdout
    assert (runs / "exp_high-mid_seed4" / "checkpoints" / "last.pt").exists()
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert again.returncode != 0 and "already holds a run" in again.stderr
    assert subprocess.run(cmd + ["--resume-auto"], capture_output=True, text=True).returncode == 0
    assert cfg["train"]["epochs"] == 1
