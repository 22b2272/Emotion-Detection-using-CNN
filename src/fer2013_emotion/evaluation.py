"""Classification metrics and plots for held-out test data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def classification_metrics(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Calculate accuracy, macro scores, and a confusion matrix from class probabilities."""
    y_true = np.asarray(true_labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    class_count = len(class_names)
    if class_count < 2:
        raise ValueError("At least two class names are required.")
    if scores.ndim != 2 or scores.shape != (len(y_true), class_count):
        raise ValueError("probabilities must have shape (number of labels, number of classes).")
    if len(y_true) == 0 or not np.isfinite(scores).all():
        raise ValueError("Labels must be non-empty and probabilities must be finite.")
    if np.any(y_true < 0) or np.any(y_true >= class_count):
        raise ValueError("true_labels contains an index outside class_names.")

    predictions = scores.argmax(axis=1)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (y_true, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision_denominator = confusion.sum(axis=0)
    recall_denominator = confusion.sum(axis=1)
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(true_positive),
        where=precision_denominator != 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(true_positive),
        where=recall_denominator != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(np.mean(predictions == y_true)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(recall_denominator[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def evaluate_model(
    model: Any,
    dataset: Any,
    class_names: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve, roc_auc_score
    except ImportError as error:
        raise ImportError(
            "scikit-learn and matplotlib are required for evaluation; install the 'ml' extra."
        ) from error

    true_labels = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    probabilities = np.asarray(model.predict(dataset, verbose=0))
    metrics = classification_metrics(true_labels, probabilities, class_names)
    one_hot = np.eye(len(class_names), dtype=np.int32)[true_labels]
    metrics["macro_roc_auc_ovr"] = float(
        roc_auc_score(true_labels, probabilities, average="macro", multi_class="ovr")
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="FER-2013 test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    for index, class_name in enumerate(class_names):
        false_positive, true_positive, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        area = auc(false_positive, true_positive)
        axis.plot(false_positive, true_positive, label=f"{class_name} (AUC = {area:.2f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="FER-2013 test ROC curves (one-vs-rest)",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path / "roc_curves.png", dpi=150)
    plt.close(figure)
    return metrics
