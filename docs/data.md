# Data: MassiveNuS convergence maps → training data

Code: `src/cosmicweb/data/` · Tests: `tests/test_{forward_model,noise,masks,io_splits,build,dataset}.py`
· Config: `configs/data.yaml` · Colab driver: `notebooks/data_pipeline.ipynb`

## 1. Get the maps

We use the MassiveNuS convergence maps (Liu et al. 2018), fiducial massless-neutrino model
`convergence_gal_mnv0.00000_om0.30000_As2.1000.tar`: 512 × 512 pixel maps, 3.5° × 3.5°,
one FITS file per line of sight (`WLconv_z<redshift>_<NNNN>r.fits`).

Put the `.tar` on the shared Drive under `Astronova/Research/DataSet/`. **Never commit data to
Git**; `.gitignore` blocks `*.fits` and `*.npy`. You do not need to extract the archive: the
pipeline reads the `.tar` directly (or an extracted folder; both give identical results).

## 2. Build the dataset (once)

```bash
python scripts/build_dataset.py --config configs/data.yaml --dataset-root /path/to/DataSet --qa high:mid
```

or run `notebooks/data_pipeline.ipynb` on Colab. For a quick trial, set
`build.max_maps: 50` in a copy of the config.

The build reads each map once, assigns train/val/test **by line of sight** (realization number),
checks for leakage and writes to `output_dir`:

| File | Contents |
|---|---|
| `kappa_{train,val,test}.npy` | float32 `(N, 512, 512)` clean κ, rows sorted by file name. Opened memory-mapped, so they are never fully loaded into RAM |
| `manifest.csv` | one row per map: `split, row, name, realization, redshift, map_key, kappa_mean, kappa_std, n_nonfinite, sha1` |
| `splits/{train,val,test}.csv` | map names per split (small; these are the "split lists" for the code release) |
| `dataset.json` | geometry, pixel area, counts, skipped maps, leakage result, build settings and a **fingerprint** of the content. Quote the fingerprint in reports to identify dataset v1 |

Safety: an existing build is never overwritten without `--overwrite`. Output is written to
`.partial/` and moved into place only when the build succeeds. Unreadable maps are skipped and
listed under `failed_maps`. A FITS `ANGLE` header that disagrees with `map_size_deg` aborts the build.

**Size:** about 1 MiB per map, roughly 10 GB for all 10,000 maps at one redshift. Only clean κ is
stored, so every noise × mask condition (task 3.4) reuses the same files.

## 3. Observations (generated on the fly)

`ShearMapDataset(root, split, cfg.observation, noise, mask, ...)` turns each κ map into:

- `x`: float32 `(3, H, W)` = `[γ₁, γ₂, mask]`, where shear is zero in masked pixels
- `y`: float32 `(1, H, W)` = κ with the full-map mean removed

Both are multiplied by `observation.scale` (10, following Jeffrey et al. 2020). The mask channel is not scaled.

Steps, always on the **full** map before any cropping:

1. **Forward model** (flat-sky Kaiser–Squires): γ̃₁ = (ℓx² − ℓy²)/ℓ² κ̃, γ̃₂ = 2ℓxℓy/ℓ² κ̃.
   Axis 0 is y and axis 1 is x. The DC mode and the Nyquist row/column are dropped. Shear is computed on the full
   periodic map; computing it on a crop would wrongly wrap the edges.
2. **Shape noise**: independent Gaussian noise per component and pixel,
   σ_pix = σ_e / √(2 n_gal A_pix), where σ_e is the *total* ellipticity dispersion and
   A_pix = (3.5° · 60 / 512)² ≈ 0.168 arcmin².
3. **Mask**: edge border, one full-width chip gap, then bright-star holes until the target
   masked fraction is reached. Sizes are in pixels; 1 px ≈ 0.41 arcmin.

Levels (tasks 1.3, 1.4) are named in the config:

| Noise | n_gal [arcmin⁻²] | σ_pix | | Mask | fraction | edge |
|---|---|---|---|---|---|---|
| `none` | – | 0 | | `none` | 0 | 0 |
| `low` | 30 (LSST-like) | 0.088 | | `low` | 5% | 4 px |
| `mid` | 15 | 0.125 | | `mid` | 15% | 8 px |
| `high` | 6 (DES SV-like) | 0.197 | | `high` | 30% | 8 px |

The noise and mask levels are **placeholders until agreed with the advisor (task 0.4).**

**Reproducibility.** Noise and masks are seeded by `observation.seed` and the map's file name,
so a given map always gets the same observation on any machine and with any file order. Noise and
mask use separate random streams, and every noise level reuses the same unit-normal draws (common
random numbers), so differences between levels come only from σ. Val/test must use
`resample_each_epoch=False` (the default) and `augment=False`. Training may use
`resample_each_epoch=True` for fresh noise and masks every epoch, and `augment=True` for random
flips and transposes. Those flip the sign of γ₂ (flips) or γ₁ (transpose), because shear is spin-2.

Exact reproducibility across machines also assumes the same NumPy version, since NumPy may change
how its random distributions are generated between releases. Record the environment with results.

**Views.** `mode="full"` gives whole maps, `"tiles"` non-overlapping `patch_size` tiles, and
`"random"` `crops_per_map` random crops per map that move with `set_epoch(epoch)`. Call
`set_epoch` at the start of every epoch, and do not combine it with `persistent_workers=True`.

## 4. Leakage

All maps of one line of sight (for example the same realization at several redshifts) go to one
split. The build also checks that no map name, realization or identical map content (SHA-1) appears in
two splits.

**Caveat:** MassiveNuS ray-traces its many realizations through one N-body simulation per
cosmology, so different lines of sight are not fully independent. Splitting by realization
removes shared maps, not all correlation. Mention this in the methods section.
