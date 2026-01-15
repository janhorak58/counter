import argparse
from typing import Iterable, Optional

from src.predict.config import load_predict_config
from src.predict.runner import run_prediction


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run prediction on a video from config.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_predict_config(args.config)
    result = run_prediction(cfg)
    return 0 if result is not None else 1
