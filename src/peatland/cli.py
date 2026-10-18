"""Command line interface: python -m peatland.cli <command>."""

from __future__ import annotations

import argparse
import time

from . import pipeline
from .config import apply_mode, load_config
from .download import run_download

COMMANDS = ["download", "preprocess", "split", "train-rf", "train-unet", "evaluate", "run-all"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="peatland", description="Peatland land-cover pipeline (Sentinel-2 + CORINE).")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--label-file", help="path to the CORINE .tif/.zip (default: labels.corine_path in config)")
    parser.add_argument("--allow-proxy", action="store_true", help="use WorldCover PROXY labels if CORINE is missing")
    parser.add_argument("--mode", choices=["demo", "full"], default="full", help="preset: demo = fast live run, full = reported run")
    parser.add_argument("--skip-unet", action="store_true", help="run-all without the U-Net")
    args = parser.parse_args(argv)
    cfg = apply_mode(load_config(args.config), args.mode)

    stages = {
        "download": lambda: run_download(cfg, args.label_file, args.allow_proxy),
        "preprocess": lambda: pipeline.stage_preprocess(cfg),
        "split": lambda: pipeline.stage_split(cfg),
        "train-rf": lambda: pipeline.stage_train_rf(cfg),
        "train-unet": lambda: pipeline.stage_train_unet(cfg),
        "evaluate": lambda: pipeline.stage_evaluate(cfg),
    }
    order = ["download", "preprocess", "split", "train-rf"] + ([] if args.skip_unet else ["train-unet"]) + ["evaluate"]
    for name in order if args.command == "run-all" else [args.command]:
        t0 = time.time()
        print(f"\n=== {name} ===")
        stages[name]()
        print(f"[{name}] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
