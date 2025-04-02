"""Data processing nodes."""

import io
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import rdata
import requests
from sklearn.model_selection import train_test_split


def download_data(url: str, output_directory: str, file_name: str) -> str:
    """Download data from a URL and save it to a CSV file.

    Args:
        url: URL to download data from
        output_directory: Directory to save the data
        file_name: Name of the file to save the data

    Returns:
        Path to the downloaded data
    """
    # Create the output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Full path to save the data
    output_path = Path(output_directory) / file_name

    # Download the file from the URL
    response = requests.get(url)

    # Write the content to a file
    f = io.BytesIO(response.content)
    r_data = rdata.read_rda(f)["pg15training"]
    r_data.to_csv(output_path, index=False)

    # Return the path as a string for the TextDataSet
    return str(output_path)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        data_path: Path to the CSV file

    Returns:
        Loaded data as a pandas DataFrame
    """
    return pd.read_csv(data_path)


def preprocess_data(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """Preprocess the data by creating a target column, dropping columns, and one-hot encoding.

    Args:
        data: Raw data
        categorical_columns: List of categorical columns to one-hot encode

    Returns:
        Preprocessed data
    """

    data["target"] = data["Numtppd"].apply(lambda x: 1 if x != 0 else 0)

    data = data.drop(columns=["Numtppd", "Numtpbi", "Indtppd", "Indtpbi"])

    # Add one hot encoder processor
    data = pd.get_dummies(data, columns=categorical_columns)

    return data


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into features and targets, training and test sets.

    Args:
        data: Preprocessed data
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training targets (as DataFrame)
        y_test: Test targets (as DataFrame)
    """
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convert Series to DataFrames for compatibility with ParquetDataSet
    y_train_df = pd.DataFrame(y_train, columns=["target"])
    y_test_df = pd.DataFrame(y_test, columns=["target"])

    return X_train, X_test, y_train_df, y_test_df
