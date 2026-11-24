"""Build the frozen clean-κ dataset from MassiveNuS maps.

    python scripts/build_dataset.py --config configs/data.yaml \
        --dataset-root /path/to/DataSet [--overwrite] [--qa high:mid]
"""

from __future__ import annotations

import argparse
import json

from cosmicweb.data import build_dataset, load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="data config YAML")
    ap.add_argument("--dataset-root", help="override dataset_root from the config")
    ap.add_argument("--overwrite", action="store_true", help="rebuild an existing dataset")
    ap.add_argument("--qa", metavar="NOISE:MASK", help="also save a QA figure for this condition, e.g. high:mid")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, dataset_root=args.dataset_root)
    print(f"source : {cfg.build.source}\noutput : {cfg.build.output_dir}")
    meta = build_dataset(cfg, overwrite=args.overwrite)
    print(json.dumps({k: meta[k] for k in ("counts", "map_shape", "pixel_area_arcmin2", "fingerprint")}, indent=2))
    if meta["failed_maps"]:
        print(f"WARNING: {len(meta['failed_maps'])} map(s) skipped; see dataset.json")

    if args.qa:
        import matplotlib

        matplotlib.use("Agg")
        from cosmicweb.data import ShearMapDataset
        from cosmicweb.data.qa import plot_examples

        noise, mask = args.qa.split(":")
        ds = ShearMapDataset(cfg.build.output_dir, "train", cfg.observation, noise, mask)
        path = cfg.build.output_dir / f"qa_{noise}_{mask}.png"
        plot_examples(ds, range(min(3, len(ds))), path)
        print(f"QA figure: {path}")


if __name__ == "__main__":
    main()
