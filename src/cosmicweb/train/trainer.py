"""Training loop with validation, checkpoints, logging and exact resume.

A run folder (``<runs_dir>/<name>/``) holds::

    config.yaml           resolved experiment config
    run.json              code version, dataset fingerprint, devices, timings, best result
    metrics.csv           one row per epoch
    tensorboard/          the same metrics for TensorBoard (if installed)
    checkpoints/last.pt   everything needed to resume
    checkpoints/best.pt   weights with the lowest validation loss (no optimizer state)

Reproducibility: the shuffle order of each epoch comes from a generator seeded
by ``(seed, epoch)``, and crops, augmentation and noise are keyed by epoch in
the dataset, so an interrupted run resumed from ``last.pt`` continues exactly
as if it had never stopped (bitwise on CPU; on GPU up to non-deterministic
CUDA kernels).
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..data import load_config
from ..data.dataset import ShearMapDataset
from ..models import UNet
from .config import run_name

__all__ = ["train", "load_model", "build_loss", "METRIC_FIELDS"]

METRIC_FIELDS = ("epoch", "train_loss", "val_loss", "val_rmse_kappa", "lr", "epoch_time_s",
                 "train_items_per_s", "peak_gpu_mem_gb")
# Settings that may differ when resuming (paths and machine-specific knobs, or a longer run).
RESUMABLE_CHANGES = {"train.epochs", "output.name", "train.early_stopping_patience", "data.num_workers", "data.root",
                     "data.dataset_root", "data.config", "output.runs_dir", "output.tensorboard"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _git_state() -> dict:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _atomic_save(obj, path: Path, attempts: int = 30) -> None:
    """Write via a temporary file so a crash never leaves a half-written checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    for i in range(attempts):  # Windows antivirus can briefly lock a fresh file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if os.name != "nt" or i == attempts - 1:
                raise
            time.sleep(min(0.05 * 2**i, 1.0))


def build_loss(cfg: dict) -> nn.Module:
    """Loss from ``cfg['loss']``. Only ``mse`` exists for now; the power-spectrum loss is #18."""
    name, extra = cfg["name"], set(cfg) - {"name"}
    if name == "mse":
        if extra:
            raise ValueError(f"loss 'mse' takes no options, got {sorted(extra)}")
        return nn.MSELoss()
    raise ValueError(f"unknown loss {name!r}; available: 'mse'")


def _build_optimizer(cfg: dict, params) -> torch.optim.Optimizer:
    o = cfg["optim"]
    if o["name"] == "adamw":
        return torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["weight_decay"])
    raise ValueError(f"unknown optimizer {o['name']!r}; available: 'adamw', 'adam'")


def _build_scheduler(cfg: dict, opt):
    s = cfg["schedule"]
    if s["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"], eta_min=s["min_lr"])
    if s["name"] == "constant":
        return None
    raise ValueError(f"unknown schedule {s['name']!r}; available: 'cosine', 'constant'")


def _build_datasets(cfg: dict):
    d = cfg["data"]
    if d["val"].get("augment") or d["val"].get("resample_each_epoch"):
        raise ValueError("validation must not use augment or resample_each_epoch: its inputs must never change")
    dcfg = load_config(d["config"], dataset_root=d["dataset_root"])
    root = Path(d["root"]).resolve() if d["root"] else dcfg.build.output_dir
    if not (root / "dataset.json").exists():
        raise FileNotFoundError(f"no built dataset in {root}; run scripts/build_dataset.py or set data.root")
    train_ds = ShearMapDataset(root, "train", dcfg.observation, d["noise"], d["mask"], **d["train"])
    val_ds = ShearMapDataset(root, "val", dcfg.observation, d["noise"], d["mask"], **d["val"])
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    return train_ds, val_ds, root, meta, dcfg.observation.scale


def _loader(ds, cfg: dict, train: bool, epoch: int, device: torch.device) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg["train"]["seed"] * 1_000_003 + epoch)  # shuffle order depends only on (seed, epoch)
    return DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=train, generator=g if train else None,
                      drop_last=train, num_workers=cfg["data"]["num_workers"],
                      pin_memory=device.type == "cuda", persistent_workers=False)


def _check_resume_config(saved: dict, new: dict) -> None:
    a, b = _flatten(saved), _flatten(new)
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    blocked = [k for k in changed if k not in RESUMABLE_CHANGES]
    if blocked:
        raise ValueError("cannot resume: the config differs from the checkpoint in "
                         + ", ".join(f"{k} ({a.get(k)!r} -> {b.get(k)!r})" for k in blocked)
                         + ". Use a new run name for a different experiment.")
    if "train.epochs" in changed:
        warnings.warn("train.epochs changed on resume: the cosine schedule now differs from an "
                      "uninterrupted run with the new length", RuntimeWarning, stacklevel=3)


def _read_metrics(path: Path, up_to_epoch: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if int(r["epoch"]) <= up_to_epoch]


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: e.g. the CUDA backward of bilinear upsampling has no deterministic version
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> UNet:
    """Rebuild the U-Net from a checkpoint (``best.pt`` or ``last.pt``) in eval mode."""
    state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    model = UNet(in_channels=3, out_channels=1, **state["config"]["model"])
    model.load_state_dict(state["model"])
    return model.eval()


def train(cfg: dict, *, resume: bool | str = False, max_epochs_this_session: int | None = None,
          device: str | torch.device | None = None, verbose: bool = True) -> dict:
    """Train the model described by ``cfg`` (from :func:`load_experiment`).

    Args:
        resume: False = the run folder must not hold a run yet; True = continue
            from ``checkpoints/last.pt``; ``"auto"`` = continue if it exists.
        max_epochs_this_session: stop after this many epochs (e.g. to fit a Colab
            session); resume later with ``resume=True``.
        device: defaults to CUDA when available.

    Returns the final ``run.json`` contents.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tc = cfg["train"]
    run_dir = Path(cfg["output"]["runs_dir"]).expanduser().resolve() / run_name(cfg)
    ckpt_dir = run_dir / "checkpoints"
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    metrics_path, info_path = run_dir / "metrics.csv", run_dir / "run.json"

    if resume == "auto":
        resume = last_path.exists()
    if resume and not last_path.exists():
        raise FileNotFoundError(f"nothing to resume: {last_path} does not exist")
    if not resume and (last_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"{run_dir} already holds a run; pass resume=True (CLI: --resume) "
                              "or choose another output.name")

    _seed_everything(tc["seed"], tc["deterministic"])
    train_ds, val_ds, data_root, data_meta, scale = _build_datasets(cfg)
    model = UNet(in_channels=3, out_channels=1, **cfg["model"]).to(device)
    loss_fn = build_loss(cfg["loss"])
    opt = _build_optimizer(cfg, model.parameters())
    sched = _build_scheduler(cfg, opt)
    amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_val, best_epoch, since_best = 0, math.inf, None, 0
    info = {"sessions": []}
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        _check_resume_config(state["config"], cfg)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if sched is not None and state["scheduler"] is not None:
            sched.load_state_dict(state["scheduler"])
            sched.T_max = tc["epochs"]
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_val, best_epoch, since_best = state["best_val_loss"], state["best_epoch"], state["epochs_since_best"]
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
    # Keep the rows of the start_epoch epochs already checkpointed (rows are numbered from 1); this
    # drops a row logged for an epoch whose checkpoint never landed.
    rows = _read_metrics(metrics_path, start_epoch)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    session = {"started_utc": _utc(), "start_epoch": start_epoch, **_git_state(), "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}
    info.update({"run_name": run_dir.name, "experiment": cfg["experiment"], "parameters": model.num_parameters(),
                 "dataset_root": str(data_root), "dataset_fingerprint": data_meta["fingerprint"],
                 "train_items_per_epoch": len(train_ds), "val_items": len(val_ds)})
    info["sessions"].append(session)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    writer = None
    if cfg["output"]["tensorboard"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(run_dir / "tensorboard"))
        except ImportError:
            warnings.warn("tensorboard is not installed; logging to metrics.csv only", RuntimeWarning, stacklevel=2)

    if verbose:
        print(f"run {run_dir}\n{model.num_parameters() / 1e6:.2f} M parameters on {device}"
              f"{' (AMP)' if amp else ''} | {len(train_ds)} train items, {len(val_ds)} val items | "
              f"noise={cfg['data']['noise']} mask={cfg['data']['mask']} seed={tc['seed']}"
              + (f" | resuming at epoch {start_epoch + 1}" if resume else ""))

    stop_reason, epochs_run = "completed", 0
    for epoch in range(start_epoch, tc["epochs"]):
        if max_epochs_this_session is not None and epochs_run >= max_epochs_this_session:
            stop_reason = "session limit"
            break
        if tc["early_stopping_patience"] is not None and since_best >= tc["early_stopping_patience"]:
            stop_reason = "early stopping"
            break
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        lr = opt.param_groups[0]["lr"]

        # --- train
        train_ds.set_epoch(epoch)
        model.train()
        total, count = 0.0, 0
        for i, batch in enumerate(_loader(train_ds, cfg, True, epoch, device)):
            if tc["limit_train_batches"] is not None and i >= tc["limit_train_batches"]:
                break
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(x)
            loss = loss_fn(pred.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss became {loss.item()} at epoch {epoch + 1}, batch {i + 1}; "
                                         "try a lower optim.lr or set train.grad_clip")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc["grad_clip"] is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
            count += len(x)
        if count == 0:
            raise RuntimeError("no training batches: lower data.batch_size or check the dataset")
        train_loss = total / count
        train_time = time.perf_counter() - t0
        if sched is not None:
            sched.step()

        # --- validate
        model.eval()
        vtotal, vcount, sse, npix = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(_loader(val_ds, cfg, False, epoch, device)):
                if tc["limit_val_batches"] is not None and i >= tc["limit_val_batches"]:
                    break
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    pred = model(x)
                pred = pred.float()
                vtotal += loss_fn(pred, y).item() * len(x)
                vcount += len(x)
                sse += F.mse_loss(pred, y, reduction="sum").item()
                npix += y.numel()
        val_loss = vtotal / vcount
        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, since_best = val_loss, epoch + 1, 0
        else:
            since_best += 1

        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
               "val_rmse_kappa": math.sqrt(sse / npix) / scale, "lr": lr,
               "epoch_time_s": round(time.perf_counter() - t0, 2),
               "train_items_per_s": round(count / train_time, 1),
               "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
                                   if device.type == "cuda" else None)}
        rows.append(row)
        _write_metrics(metrics_path, rows)
        if writer is not None:
            for k in METRIC_FIELDS[1:]:
                if row[k] is not None:
                    writer.add_scalar(k, row[k], epoch + 1)
            writer.flush()

        state = {"format_version": 1, "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict() if sched is not None else None, "scaler": scaler.state_dict(),
                 "best_val_loss": best_val, "best_epoch": best_epoch, "epochs_since_best": since_best,
                 "config": cfg}
        _atomic_save(state, last_path)  # saved after metrics: a crash in between re-runs this epoch
        if improved:  # weights only (about 1/3 the size): best.pt is for evaluation, last.pt for resuming
            _atomic_save({k: state[k] for k in ("format_version", "epoch", "model", "best_val_loss", "best_epoch",
                                                "config")}, best_path)
        epochs_run += 1
        if verbose:
            print(f"epoch {epoch + 1:>3}/{tc['epochs']}  train {train_loss:.5f}  val {val_loss:.5f}"
                  f"{' *' if improved else '  '}  rmse_kappa {row['val_rmse_kappa']:.5f}  lr {lr:.2e}  "
                  f"{row['epoch_time_s']:.0f}s"
                  + (f"  peak {row['peak_gpu_mem_gb']:.2f} GB" if row["peak_gpu_mem_gb"] is not None else ""))

    if writer is not None:
        writer.close()
    session.update({"finished_utc": _utc(), "end_epoch": len(rows), "stop_reason": stop_reason})
    peaks = [float(r["peak_gpu_mem_gb"]) for r in rows if r.get("peak_gpu_mem_gb") not in (None, "")]
    info.update({"epochs_completed": len(rows), "stop_reason": stop_reason,
                 "best_val_loss": best_val if math.isfinite(best_val) else None,
                 "best_epoch": best_epoch, "peak_gpu_mem_gb": max(peaks) if peaks else None,
                 "checkpoint_mb": round(last_path.stat().st_size / 1e6, 1) if last_path.exists() else None})
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    if verbose:
        print(f"stopped: {stop_reason} after {len(rows)} epoch(s); best val {best_val:.5f} at epoch {best_epoch}")
    return info
