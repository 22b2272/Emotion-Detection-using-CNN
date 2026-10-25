"""Training and experiment artifact management."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import create_datasets, validate_dataset
from .models import build_model


def train_experiment(
    config: ExperimentConfig,
    data_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("TensorFlow is required; install the 'ml' extra.") from error

    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    dataset_paths = validate_dataset(data_dir)
    datasets = create_datasets(dataset_paths, config)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model = build_model(config, num_classes=len(datasets.class_names))
    model_path = output_path / "model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(1, config.early_stopping_patience // 2),
            min_delta=1e-4,
        ),
        tf.keras.callbacks.CSVLogger(str(output_path / "training.csv")),
    ]

    fit_options: dict[str, Any] = {}
    if config.use_class_weights:
        total = sum(datasets.train_class_counts)
        class_count = len(datasets.train_class_counts)
        fit_options["class_weight"] = {
            index: total / (class_count * count)
            for index, count in enumerate(datasets.train_class_counts)
        }

    history = model.fit(
        datasets.train,
        epochs=config.epochs,
        validation_data=datasets.validation,
        callbacks=callbacks,
        **fit_options,
    )
    if not model_path.is_file():
        raise RuntimeError("Training finished without producing the expected best-model checkpoint.")

    metadata = {
        "variant": config.variant,
        "class_names": list(datasets.class_names),
        "config": asdict(config),
        "epochs_completed": len(history.epoch),
        "best_validation_loss": min(history.history["val_loss"]),
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return model_path
