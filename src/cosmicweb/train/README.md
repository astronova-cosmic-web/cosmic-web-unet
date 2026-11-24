# train

Training loop, checkpointing and logging.

```bash
python scripts/train.py --config configs/baseline.yaml --set train.seed=1 --set data.noise=low
python scripts/train.py --config configs/baseline.yaml --resume          # continue after an interruption
```

On Colab, use `notebooks/train.ipynb`. It copies dataset v1 to local disk, writes runs to Drive, and resumes automatically when re-run.

## Configs

There is one YAML file per experiment (for example `configs/baseline.yaml`). Every key has a default in `config.py`, and unknown keys are errors. `--set key.sub=value` overrides one setting. Values are read as YAML, and numbers such as `1e-4` are accepted.

| Section | What it sets |
|---|---|
| `data` | data config, noise/mask level, train and val views (`mode`, `patch_size`, `augment`, ...), batch size, workers |
| `model` | U-Net width, depth, norm, padding |
| `loss` | `{name: mse}`; the power-spectrum loss (#18) will add options here |
| `optim`, `schedule` | AdamW or Adam; cosine decay to `min_lr` or a constant learning rate |
| `train` | epochs, seed, mixed precision, gradient clipping, early stopping, batch limits for pilots |
| `output` | `runs_dir`, run name (default `<experiment>_<noise>-<mask>_seed<seed>`), TensorBoard |

## Run folder

```
<runs_dir>/<run name>/
  config.yaml          the resolved config
  run.json             code commit, dataset fingerprint, GPU, sessions, best epoch, peak GPU memory, checkpoint size
  metrics.csv          per epoch: train/val loss, val RMSE in κ units, lr, time, items/s, peak GPU memory
  tensorboard/
  checkpoints/last.pt  model, optimizer, scheduler and AMP state, for resuming
  checkpoints/best.pt  weights with the lowest val loss, for evaluation (cosmicweb.train.load_model)
```

## Reproducibility

- **Data order:** each epoch's shuffle order depends only on `(seed, epoch)`. Crops, augmentation and noise are keyed by epoch in the dataset. So a run resumed from `last.pt` matches an uninterrupted run exactly on CPU; this is tested.
- **GPU:** results can differ in the last digits, because some CUDA kernels (for example the backward pass of bilinear upsampling) are not deterministic.
- **Validation:** the validation inputs are fixed (no augmentation and no resampling, enforced), so validation losses can be compared across epochs, seeds and runs.
- **Resume safety:** resuming refuses a changed experiment (for example a different noise level). Only paths, worker count, run length and patience may change.
