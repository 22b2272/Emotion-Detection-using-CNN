"""Command-line interface for FER-2013 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import VARIANTS, make_config
from .data import create_datasets, validate_dataset
from .evaluation import evaluate_model
from .inference import load_trained_model, predict_image
from .training import train_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate FER-2013 emotion classifiers.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train one model variant.")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--data", type=Path, default=Path("data"))
    train.add_argument("--output", type=Path, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="Evaluate a saved model on the held-out test split.")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=Path("data"))
    evaluate.add_argument("--output", type=Path, default=None)

    predict = commands.add_parser("predict", help="Predict the emotion for one image.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        config = make_config(
            args.variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        output = args.output or Path("outputs") / args.variant
        model_path = train_experiment(config, args.data, output)
        print(f"Best model saved to {model_path}")
        print(f"Run test evaluation separately with --model {model_path}")
        return 0

    if args.command == "evaluate":
        model, metadata = load_trained_model(args.model)
        dataset_paths = validate_dataset(args.data)
        if list(dataset_paths.class_names) != metadata.get("class_names"):
            raise ValueError("Dataset class names/order do not match the saved model metadata.")
        config = make_config(metadata["variant"])
        datasets = create_datasets(dataset_paths, config)
        output = args.output or args.model.parent / "evaluation"
        results = evaluate_model(model, datasets.test, datasets.class_names, output)
        print(json.dumps(results, indent=2))
        print(f"Evaluation artifacts saved to {output}")
        return 0

    model, metadata = load_trained_model(args.model)
    print(json.dumps(predict_image(model, metadata, args.image), indent=2))
    return 0
