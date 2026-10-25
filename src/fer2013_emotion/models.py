"""Model builders for the implemented FER-2013 experiments."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig


def build_model(config: ExperimentConfig, num_classes: int) -> Any:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("TensorFlow is required; install the 'ml' extra.") from error

    layers = tf.keras.layers
    inputs = tf.keras.Input(
        shape=(config.image_size, config.image_size, config.channels),
        name="face_image",
    )

    if config.variant == "resnet50v2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="resnet_preprocessing")(inputs)
    else:
        x = layers.Rescaling(1.0 / 255.0, name="pixel_scaling")(inputs)

    if config.use_augmentation:
        x = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(40.0 / 360.0, fill_mode="nearest"),
                layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
                layers.RandomZoom(0.2, fill_mode="nearest"),
            ],
            name="training_augmentation",
        )(x)

    if config.variant == "resnet50v2":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(config.image_size, config.image_size, config.channels),
        )
        x = base_model(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        for units in (512, 256, 128):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(0.1)(x)
    else:
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(256, 3, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(512, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(512, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.variant.replace("-", "_"))
    optimizer = (
        tf.keras.optimizers.Adamax(learning_rate=config.learning_rate)
        if config.variant == "resnet50v2"
        else tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
