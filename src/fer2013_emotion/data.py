"""FER-2013 dataset validation, splitting, and TensorFlow input pipelines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train_dir: Path
    test_dir: Path
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetBundle:
    class_names: tuple[str, ...]
    train: Any
    validation: Any
    test: Any
    train_class_counts: tuple[int, ...]


def _class_directories(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset split directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for child in directory.iterdir():
        if child.is_dir():
            key = child.name.casefold()
            if key in result:
                raise ValueError(f"Duplicate class directory name (case-insensitive): {child.name}")
            result[key] = child
    if not result:
        raise ValueError(f"No class subdirectories found in {directory}")
    return result


def validate_dataset(data_dir: str | Path, expected_classes: int = 7) -> DatasetPaths:
    """Validate the expected train/<class> and test/<class> image-tree without modifying it."""
    root = Path(data_dir).expanduser().resolve()
    train_dir = root / "train"
    test_dir = root / "test"
    train_classes = _class_directories(train_dir)
    test_classes = _class_directories(test_dir)

    if set(train_classes) != set(test_classes):
        missing_test = sorted(set(train_classes) - set(test_classes))
        missing_train = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            f"Train/test classes do not match; missing from test: {missing_test}; "
            f"missing from train: {missing_train}."
        )
    if len(train_classes) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(train_classes)}.")

    class_names = tuple(sorted((path.name for path in train_classes.values()), key=str.casefold))
    corrupt: list[str] = []
    for split_name, classes in (("train", train_classes), ("test", test_classes)):
        for class_dir in classes.values():
            images = sorted(path for path in class_dir.rglob("*") if path.is_file())
            if split_name == "train" and len(images) < 2:
                raise ValueError(f"Class {class_dir.name!r} needs at least two training images.")
            if split_name == "test" and not images:
                raise ValueError(f"Class {class_dir.name!r} has no test images.")
            for image_path in images:
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    corrupt.append(f"{image_path} (unsupported file type)")
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    corrupt.append(f"{image_path} ({error})")
    if corrupt:
        examples = "; ".join(corrupt[:5])
        extra = f" and {len(corrupt) - 5} more" if len(corrupt) > 5 else ""
        raise ValueError(f"Dataset contains invalid image files: {examples}{extra}")

    return DatasetPaths(root, train_dir, test_dir, class_names)


def stratified_split(
    class_image_paths: Sequence[Sequence[Path]],
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[Path, ...], tuple[int, ...], tuple[Path, ...], tuple[int, ...]]:
    """Make a deterministic, per-class train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    validation_paths: list[Path] = []
    validation_labels: list[int] = []
    for label, paths in enumerate(class_image_paths):
        ordered = sorted(paths)
        if len(ordered) < 2:
            raise ValueError(f"Class index {label} needs at least two training images to split.")
        rng.shuffle(ordered)
        validation_count = min(max(round(len(ordered) * validation_fraction), 1), len(ordered) - 1)
        validation = ordered[:validation_count]
        training = ordered[validation_count:]
        train_paths.extend(training)
        train_labels.extend([label] * len(training))
        validation_paths.extend(validation)
        validation_labels.extend([label] * len(validation))

    return (
        tuple(train_paths),
        tuple(train_labels),
        tuple(validation_paths),
        tuple(validation_labels),
    )


def create_datasets(paths: DatasetPaths, config: Any) -> DatasetBundle:
    """Create reproducible TensorFlow datasets; augmentation is applied in the training model only."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("TensorFlow and NumPy are required; install the 'ml' extra.") from error

    class_dirs = {child.name.casefold(): child for child in paths.train_dir.iterdir() if child.is_dir()}
    class_image_paths = [
        [
            path
            for path in class_dirs[class_name.casefold()].rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        for class_name in paths.class_names
    ]
    train_paths, train_labels, validation_paths, validation_labels = stratified_split(
        class_image_paths, config.validation_fraction, config.seed
    )

    test_class_dirs = {
        child.name.casefold(): child for child in paths.test_dir.iterdir() if child.is_dir()
    }
    test_paths: list[Path] = []
    test_labels: list[int] = []
    for label, class_name in enumerate(paths.class_names):
        class_files = sorted(
            path
            for path in test_class_dirs[class_name.casefold()].rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        test_paths.extend(class_files)
        test_labels.extend([label] * len(class_files))

    def make_dataset(image_paths: Sequence[Path], labels: Sequence[int], training: bool) -> Any:
        dataset = tf.data.Dataset.from_tensor_slices(
            ([str(path) for path in image_paths], np.asarray(labels, dtype=np.int32))
        )

        def load_image(path: Any, label: Any) -> tuple[Any, Any]:
            encoded = tf.io.read_file(path)
            image = tf.image.decode_image(encoded, channels=config.channels, expand_animations=False)
            image.set_shape((None, None, config.channels))
            image = tf.image.resize(image, (config.image_size, config.image_size))
            return tf.cast(image, tf.float32), label

        if training:
            dataset = dataset.shuffle(
                len(image_paths), seed=config.seed, reshuffle_each_iteration=True
            )
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(config.batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)

    return DatasetBundle(
        class_names=paths.class_names,
        train=make_dataset(train_paths, train_labels, training=True),
        validation=make_dataset(validation_paths, validation_labels, training=False),
        test=make_dataset(test_paths, test_labels, training=False),
        train_class_counts=tuple(
            sum(label == class_index for label in train_labels)
            for class_index in range(len(class_image_paths))
        ),
    )
