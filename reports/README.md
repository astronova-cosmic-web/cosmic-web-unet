# reports

Milestone reports, QA figures, and presentation material.

## Where Results Live

To keep the repository fast and clean while making results easy for the team to inspect:

| What | Where | Why |
|---|---|---|
| Raw `.tar`, built `.npy` datasets (`dataset_v1`), model checkpoints | Google Drive (`Astronova/Research/DataSet/`) | Too large for Git (~10 GB per build) |
| Per-epoch training curves, validation metrics, sample prediction images | Weights & Biases (W&B) | Live browser dashboard for frequent runs (Phase 2+) |
| Key milestone summaries & QA figures | `reports/` (this folder, in GitHub) | Renders inline on GitHub for team & advisor review |

> Keep `.ipynb` notebooks in Git free of heavy embedded image outputs to avoid large diffs and merge conflicts.
