# Deep Learning the Cosmic Web

Most of the matter in the universe is dark matter, which we cannot see directly but can map through
weak gravitational lensing. This project reconstructs 2D convergence (mass) maps from simulated,
noisy and masked shear data using a compact U-Net. We compare a standard pixel-loss (MSE) baseline
against a hybrid loss that adds a power-spectrum term, to test whether it better preserves the
statistics cosmologists measure from these maps. AstroNova 2026 student research project.

## Research question

How effectively can a U-Net reconstruct weak-lensing mass maps as an image-to-image task, and does
adding a power-spectrum loss preserve the statistics that pixel loss alone loses? Tested against a
pixel-loss-only U-Net across controlled noise and masking conditions.

## Repository layout

| Path | Contents |
|---|---|
| `src/cosmicweb/data/` | Forward model (κ → γ₁, γ₂), shape noise, masks, line-of-sight splits |
| `src/cosmicweb/models/` | Compact U-Net |
| `src/cosmicweb/losses/` | Pixel loss and power-spectrum loss |
| `src/cosmicweb/metrics/` | RMSE, SSIM, power spectrum, peak counts |
| `src/cosmicweb/train/` | Training loop and utilities |
| `configs/` | Experiment configuration files |
| `notebooks/` | Exploration and figures |
| `scripts/` | Command-line helpers |
| `tests/` | Automated tests |
| `docs/`, `reports/` | Documentation, reports and figures |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest
```

On Colab or Kaggle, do **not** reinstall `torch` or `numpy`; use the preinstalled versions and install only what is missing.

## Data

Data is **not** stored in this repository. See `docs/` for download instructions.

- MassiveNuS convergence maps (Liu et al. 2018), the ground truth for training and testing
- DES Y3 shape catalogue (Gatti et al. 2021), used only if the Week 33 feasibility gate passes
- DESI cluster and void catalogues, used only if the Week 33 feasibility gate passes

## How to cite

Citation details will be added at the code release (Week 43).

## Advisor

Dr. Prasad Maddumage
