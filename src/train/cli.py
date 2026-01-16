import argparse
from typing import Iterable, Optional

from src.train.config import load_train_config
from src.train.runner import run_training


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train detector model from config.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_train_config(args.config)
    run_training(cfg)
    return 0
