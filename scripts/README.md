# scripts

Command-line entry points.

| Script | Purpose |
|---|---|
| `build_dataset.py` | Build the frozen κ dataset: `python scripts/build_dataset.py --config configs/data.yaml --dataset-root /path/to/DataSet` |
| `train.py` | Train a model: `python scripts/train.py --config configs/baseline.yaml [--set train.seed=1] [--resume]` |
