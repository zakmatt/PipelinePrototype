# Insurance Prediction Project (Kedro)

## Overview

This project implements a machine learning pipeline to predict insurance charges based on individual characteristics. It refactors an initial prototype notebook (`notebooks/modeling_starter.ipynb`) into a robust, maintainable, and reproducible pipeline using the [Kedro](https://kedro.org/) framework.

The pipeline performs the following steps:
1.  Downloads insurance data.
2.  Preprocesses the data (handling types, potential cleaning).
3.  Splits data into training and testing sets.
4.  Tunes hyperparameters for a LightGBM classifier using Optuna.
5.  Trains the final LightGBM model using the best hyperparameters.
6.  Evaluates the model on the test set, generating metrics and a confusion matrix.

## Project Structure

This project follows the standard Kedro layout:

```
├── conf/             # Configuration files (parameters, data catalog)
├── data/             # Project data (raw, intermediate, processed, models, reporting)
├── notebooks/        # Jupyter notebooks (e.g., exploratory analysis, original prototype)
├── src/              # Project source code
│   ├── insurance_prediction/  # Python package for the pipeline
│   │   ├── pipelines/       # Pipeline definitions (data_processing, model_training, etc.)
│   │   └── __init__.py, settings.py, etc.
│   └── tests/             # Unit and integration tests
├── .pre-commit-config.yaml # Pre-commit hook configurations
├── Dockerfile          # Docker configuration for containerization
├── pyproject.toml      # Project metadata and development dependencies
└── requirements.txt    # Runtime dependencies
```

## Setup Instructions

### Prerequisites

*   Python 3.11+
*   Conda or venv/virtualenv for environment management
*   Docker (for containerized execution)
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    *Using venv:*
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    *Or using Conda:*
    ```bash
    # It's recommended to create an environment.yml file for conda
    conda create --name insurance_pred python=3.11
    conda activate insurance_pred
    ```

3.  **Install dependencies:**
    Install both runtime dependencies (`requirements.txt`) and development dependencies (`pyproject.toml`) including the project package itself in editable mode:
    ```bash
    pip install -r requirements.txt
    pip install -e '.[dev]'
    ```

4.  **Set up pre-commit hooks:**
    Install the git hooks defined in `.pre-commit-config.yaml` to automatically check and format code before commits:
    ```bash
    pre-commit install
    ```

## Usage

### Running the Pipeline (Local)

Execute the complete Kedro pipeline:

```bash
kedro run
```

You can also run specific pipelines by name:

```bash
kedro run --pipeline=dp # Run only data_processing
kedro run --pipeline=mt # Run only model_training
kedro run --pipeline=me # Run only model_evaluation
```

### Running Tests

Execute the test suite using pytest:

```bash
pytest
```

To run tests with coverage reporting:

```bash
pytest --cov=insurance_prediction src/tests
```

### Running with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t insurance_prediction .
    ```

2.  **Run the pipeline inside the container:**
    This command runs the default pipeline (`kedro run`) inside the container. It mounts your local `data` and `logs` directories so the container can access inputs and write outputs back to your host machine.
    ```bash
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs insurance_prediction
    ```
    *Note for Windows Users:* You might need to adjust the volume paths depending on your shell (e.g., use `${pwd}` in PowerShell or absolute paths).

    You can also pass arguments to `kedro` inside the container:
    ```bash
    # Run only the data processing pipeline via Docker
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs insurance_prediction run --pipeline=dp
    ```

## Configuration

Key configurations are managed in the `conf/base` directory:

*   `parameters.yml`: Defines parameters used in the pipeline, such as `random_state`, Optuna trial counts (`n_trials`), data URLs, etc. These can be easily modified without changing the code.
*   `catalog.yml`: Defines how data is loaded and saved (datasets). It specifies file paths, file formats (like CSV, Parquet, JSON), and any specific load/save arguments.


## Testing Strategy

*   **Unit Tests** (`src/tests/unit`): Verify the functionality of individual nodes (functions) in isolation. Mock data is used where appropriate.
*   **Integration Tests** (`src/tests/integration`): Ensure that pipelines can be created and potentially run together correctly (current tests check creation).
*   **Coverage**: Test coverage is measured using `pytest-cov`. The target coverage is >= 50% (currently meets this target).

## Code Quality

Code quality and formatting are enforced using pre-commit hooks configured in `.pre-commit-config.yaml`. This includes:

*   `black`: Uncompromising code formatting.
*   `isort`: Import sorting.
*   `ruff`: Fast linting and fixing (replaces Flake8, pylint, etc.).
*   Basic checks for whitespace, file endings, etc.
