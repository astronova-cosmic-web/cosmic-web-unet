# Phase 1: Data Acquisition & Pre-processing Pipeline

Open [`phase1_data_pipeline.ipynb`](./phase1_data_pipeline.ipynb) in Google Colab to run the weak lensing data acquisition, Kaiser–Squires shear forward modeling, survey noise injection, normalization, and realization-based dataset splitting following the methodologies of Liu et al. (2018) and Jeffrey et al. (2020).

The pipeline uses a unified YAML configuration, Google Drive auto-mounting for Colab, and realization-safe manifest exports.

## Steps Implemented (Strictly Phase 1)
- **Step 1.1: Bulk Map Ingestion:** Resolves local or Google Drive paths (`/content/drive/MyDrive/Astronova/Research/DataSet`), recursively discovers MassiveNuS Model 00 512×512 FITS maps, and parses realization IDs (`_XXXXr.fits`).
- **Step 1.2: Shape Noise Injection:** Converts convergence $\kappa_{\text{true}} \to (\gamma_1, \gamma_2)$ via exact flat-sky Kaiser–Squires Fourier kernels on `rfft2`, calculates survey noise $\sigma_{\text{pixel}} = \sigma_\epsilon / \sqrt{2 n_{\text{gal}} A_{\text{pixel}}}$, and injects independent Gaussian shape noise.
- **Step 1.3: Normalization & Tensor Preparation:** Applies global scaling ($10\times$) and formats PyTorch standard tensors $X \in \mathbb{R}^{B \times C \times 512 \times 512}$ and $Y \in \mathbb{R}^{B \times 1 \times 512 \times 512}$.
- **Step 1.4: Dataset Splitting:** Partitions maps strictly by line-of-sight realization into Training (70%), Validation (15%), and Testing (15%) subsets with zero data leakage, exporting manifests.
- **Visual QA:** Generates publication-quality comparison figure verifying tensor channels against ground truth.

## Output Layout

Outputs are saved under `OUTPUT_DIR` (default: `generated_data/phase1_sample_1`):
```text
generated_data/phase1_sample_1/
├── splits/
│   ├── train_manifest.csv        # 70% simulation realizations
│   ├── val_manifest.csv          # 15% simulation realizations
│   └── test_manifest.csv         # 15% simulation realizations
├── manifest.csv                  # Full catalogue of processed source maps
├── qa_phase1_preprocessing.png   # Visual QA figure
└── config_used.yaml              # Resolved configuration
```

