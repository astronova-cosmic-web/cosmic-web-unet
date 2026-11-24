"""Train a model from an experiment config.

    python scripts/train.py --config configs/baseline.yaml \
        [--set train.seed=1 --set data.noise=low ...] [--runs-dir DIR] [--data-root DIR] \
        [--resume | --resume-auto] [--max-epochs-this-session N]
"""

from __future__ import annotations

import argparse

from cosmicweb.train import load_experiment, train


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="experiment YAML, e.g. configs/baseline.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting, e.g. train.seed=1 (repeatable)")
    ap.add_argument("--runs-dir", help="where run folders go (overrides output.runs_dir)")
    ap.add_argument("--data-root", help="built dataset folder (overrides data.root)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="continue the run from checkpoints/last.pt")
    group.add_argument("--resume-auto", action="store_true", help="continue if a checkpoint exists, else start")
    ap.add_argument("--max-epochs-this-session", type=int, help="stop after N epochs; resume later")
    ap.add_argument("--device", help="e.g. cuda, cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    cfg = load_experiment(args.config, args.overrides)
    if args.runs_dir:  # set directly: paths are not parsed as YAML
        cfg["output"]["runs_dir"] = args.runs_dir
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    train(cfg, resume="auto" if args.resume_auto else args.resume,
          max_epochs_this_session=args.max_epochs_this_session, device=args.device)


if __name__ == "__main__":
    main()
