"""Single-image FER-2013 inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import make_config


def load_trained_model(model_path: str | Path) -> tuple[Any, dict[str, Any]]:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("TensorFlow is required; install the 'ml' extra.") from error

    path = Path(model_path).expanduser().resolve()
    metadata_path = path.parent / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata file does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return tf.keras.models.load_model(path, compile=False), metadata


def predict_image(model: Any, metadata: dict[str, Any], image_path: str | Path) -> dict[str, Any]:
    try:
        config = make_config(metadata["variant"])
        class_names = metadata["class_names"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Model metadata is incomplete or invalid.") from error

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    mode = "L" if config.channels == 1 else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode).resize((config.image_size, config.image_size))
        pixels = np.asarray(image, dtype=np.float32)
    if config.channels == 1:
        pixels = pixels[..., np.newaxis]
    probabilities = np.asarray(model.predict(pixels[np.newaxis, ...], verbose=0))[0]
    if probabilities.shape != (len(class_names),):
        raise ValueError("Model output size does not match the class names in metadata.")
    predicted_index = int(probabilities.argmax())
    return {
        "class_name": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(class_names)
        },
    }
