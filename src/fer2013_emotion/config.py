"""Shared experiment configuration."""

from dataclasses import dataclass

VARIANTS = ("custom-cnn", "custom-cnn-augmented", "resnet50v2")


@dataclass(frozen=True)
class ExperimentConfig:
    variant: str
    image_size: int
    channels: int
    batch_size: int
    epochs: int
    learning_rate: float
    validation_fraction: float
    seed: int
    use_augmentation: bool
    use_class_weights: bool
    early_stopping_patience: int


def make_config(
    variant: str,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> ExperimentConfig:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {', '.join(VARIANTS)}.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    is_resnet = variant == "resnet50v2"
    defaults = {
        "image_size": 224 if is_resnet else 48,
        "channels": 3 if is_resnet else 1,
        "batch_size": 32 if is_resnet else 64,
        "epochs": 50 if is_resnet else 30,
        "learning_rate": 1e-4,
        "use_augmentation": variant != "custom-cnn",
        "use_class_weights": is_resnet,
        "early_stopping_patience": 6 if is_resnet else 5,
    }
    return ExperimentConfig(
        variant=variant,
        image_size=defaults["image_size"],
        channels=defaults["channels"],
        batch_size=batch_size or defaults["batch_size"],
        epochs=epochs or defaults["epochs"],
        learning_rate=defaults["learning_rate"],
        validation_fraction=validation_fraction,
        seed=seed,
        use_augmentation=defaults["use_augmentation"],
        use_class_weights=defaults["use_class_weights"],
        early_stopping_patience=defaults["early_stopping_patience"],
    )
