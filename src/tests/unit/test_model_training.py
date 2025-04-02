"""Unit tests for the model training nodes."""

import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from insurance_prediction.pipelines.model_training.nodes import (
    train_model,
    tune_model_hyperparameters,
)


@pytest.fixture
def sample_data() -> (pd.DataFrame, pd.DataFrame):
    """Create sample data for testing."""
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    return X, y


def test_train_model(sample_data):
    """Test the train_model function."""
    X_train, y_train = sample_data
    best_params = {
        "objective": "binary",
        "n_jobs": -1,
        "n_estimators": 50,
        "learning_rate": 0.1,
    }
    random_state = 42  # Define a random state for the test

    # Pass random_state to train_model
    model = train_model(X_train, y_train, best_params, random_state)

    assert isinstance(model, LGBMClassifier)
    # Check if the model appears to be fitted (LGBM uses _Booster attribute)
    assert hasattr(model, "_Booster") and model._Booster is not None
    assert model.get_params()["n_estimators"] == 50  # Check a param


def test_tune_model_hyperparameters(sample_data):
    """Test the tune_model_hyperparameters function."""
    X_train, y_train = sample_data
    n_trials = 3  # Use a small number of trials for testing
    random_state = 42

    best_params = tune_model_hyperparameters(X_train, y_train, n_trials, random_state)

    assert isinstance(best_params, dict)
    # Check for some expected hyperparameter keys found by Optuna
    assert "n_estimators" in best_params
    assert "learning_rate" in best_params
    # Check for fixed keys added after tuning
    assert best_params["objective"] == "binary"
    assert best_params["n_jobs"] == -1
