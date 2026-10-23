# Full-map forward-model workflow

Open [`forward_model_pipeline.ipynb`](./forward_model_pipeline.ipynb) in Google
Colab to generate noisy, masked full-map data from each FITS source map. The
notebook supports full maps, fixed tiles, and reproducible epoch-varying random
crops from the same stored observations.

## Configure for your Drive

1. Mount Google Drive in the Colab runtime.
2. Copy [`../../configs/forward_model/example.yaml`](../../configs/forward_model/example.yaml)
   to a YAML file in your Drive, such as
   `MyDrive/Astronova/Research/DataSet/config_forward_model.yaml`.
3. Edit `DATASET_ROOT`, `INPUT_DIR`, and `OUTPUT_DIR` in that copy. Input and
   output paths may be absolute, or relative to `DATASET_ROOT`.
4. Set `FORWARD_MODEL_CONFIG` to the config file path before running the
   notebook, or set `CONFIG_PATH_OVERRIDE` in section 1.

The built-in Colab defaults point to the sample `Subset` folder and write to
`generated_data/sample_1`. That output folder contains:

```text
generated_data/sample_1/
├── processed_maps.tar       # generated *_input.npy and *_target.npy arrays
├── manifest.csv              # one row per successfully processed FITS map
├── qa_forward_model.png      # visual QA figure
└── config_used.yaml          # resolved settings for this run
```

The original input TAR is left untouched. It is not read directly: extract its
FITS files first, then set `INPUT_DIR` to the extracted sample folder. Change
`INPUT_DIR` to the extracted full dataset and `OUTPUT_DIR` to a different
generated-data subfolder when you process the full dataset. All generated
artifacts stay on Drive, not in the Git repository.

For the full parameter list and crop/data-loader behavior, see the notebook's
configuration and section 6.
