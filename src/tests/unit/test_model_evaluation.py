"""Unit tests for the model evaluation nodes."""

from pathlib import Path

import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from insurance_prediction.pipelines.model_evaluation.nodes import (
    evaluate_model,
    plot_confusion_matrix,
)


@pytest.fixture
def sample_test_data() -> (pd.DataFrame, pd.DataFrame):
    """Create sample test data."""
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 5, 6, 9, 10],
            "feature2": [10, 9, 6, 5, 2, 1],
        }
    )
    y = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
    return X, y


@pytest.fixture
def mock_trained_model(sample_test_data) -> LGBMClassifier:
    """Create a simple, minimally trained mock model."""
    X, y = sample_test_data
    # Use only a tiny subset for faster fitting in test
    model = LGBMClassifier(n_estimators=2, random_state=42, verbosity=-1)
    model.fit(X.iloc[:2], y.iloc[:2]["target"].values)
    return model


def test_evaluate_model(mock_trained_model, sample_test_data):
    """Test the evaluate_model function."""
    X_test, y_test = sample_test_data
    metrics = evaluate_model(mock_trained_model, X_test, y_test)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    # Check types (optional but good practice)
    assert isinstance(metrics["accuracy"], float)


def test_plot_confusion_matrix(mock_trained_model, sample_test_data, tmp_path):
    """Test the plot_confusion_matrix function."""
    X_test, y_test = sample_test_data
    output_dir = str(tmp_path)

    # Ensure the target subdirectory does not exist initially
    plot_path = Path(output_dir) / "plots" / "confusion_matrix.png"
    assert not plot_path.exists()

    # Call the function
    plot_confusion_matrix(mock_trained_model, X_test, y_test, output_dir)

    # Check if the plot file was created
    assert plot_path.exists()
    assert plot_path.is_file()
