"""Model evaluation nodes."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate model performance on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets (DataFrame)

    Returns:
        Dictionary of model metrics
    """

    y_test_values = y_test["target"].values

    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test_values, y_pred),
        "precision": precision_score(y_test_values, y_pred),
        "recall": recall_score(y_test_values, y_pred),
        "f1_score": f1_score(y_test_values, y_pred),
    }

    return metrics


def plot_confusion_matrix(
    model, X_test: pd.DataFrame, y_test: pd.DataFrame, output_directory: str
) -> None:
    """Plot and save confusion matrix.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets (DataFrame)
        output_directory: Directory to save the plot
    """

    y_test_values = y_test["target"].values

    y_pred = model.predict(X_test)

    # Create plot directory if it doesn't exist
    plot_dir = Path(output_directory) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_test_values, y_pred)
    plt.title("Confusion Matrix")

    plt.savefig(plot_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
