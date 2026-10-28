import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from configs.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    THRESHOLD,
)


def evaluate_model(
    model,
    dataset,
    true_labels,
    metrics_path=METRICS_PATH,
    confusion_matrix_path=CONFUSION_MATRIX_PATH,
):
    probabilities = model.predict(dataset, verbose=1).ravel()
    predictions = (probabilities >= THRESHOLD).astype(np.int32)
    matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])
    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "threshold": THRESHOLD,
        "roc_auc": float(roc_auc_score(true_labels, probabilities)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion_matrix_path = Path(confusion_matrix_path)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="ChestScan AI - Test Confusion Matrix",
    )
    figure.colorbar(image, ax=axis)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(confusion_matrix_path)
    plt.close(figure)

    return metrics, probabilities, predictions
