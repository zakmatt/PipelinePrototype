"""Model training nodes."""

from typing import Any, Dict

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def objective(trial, X_train, y_train, random_state):
    """Objective function for Optuna hyperparameter tuning.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training targets (DataFrame)
        random_state: Random seed for reproducibility

    Returns:
        Accuracy score on validation set
    """
    # Split the data into training and validation sets
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    # Extract target values as array
    y_train_values = y_train_sub["target"].values
    y_val_values = y_val["target"].values

    # Define the hyperparameter search space
    param = {
        "objective": "binary",
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    # Initialize the model with the chosen set of hyperparameters
    model = LGBMClassifier(**param, verbosity=-1)

    # Train the model
    model.fit(X_train_sub, y_train_values)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val_values, y_pred)

    return accuracy


def tune_model_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.DataFrame, n_trials: int, random_state: int
) -> Dict[str, Any]:
    """Tune model hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training targets (DataFrame)
        n_trials: Number of trials for hyperparameter optimization
        random_state: Random seed for reproducibility

    Returns:
        Best hyperparameters
    """
    # Setting the logging level WARNING, the INFO logs are suppressed
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, random_state),
        n_trials=n_trials,
    )

    # Retrieve the best hyperparameters
    best_params = study.best_params

    # Add fixed parameters
    best_params.update(
        {
            "objective": "binary",
            "n_jobs": -1,
        }
    )

    return best_params


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, best_params: Dict[str, Any]
) -> LGBMClassifier:
    """Train a model with the best hyperparameters.

    Args:
        X_train: Training features
        y_train: Training targets (DataFrame)
        best_params: Best hyperparameters from tuning

    Returns:
        Trained model
    """
    # Extract target values as array
    y_train_values = y_train["target"].values

    # Train the final model with the best hyperparameters
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train_values)

    return model
