"""Model training pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from insurance_prediction.pipelines.model_training.nodes import (
    tune_model_hyperparameters,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline.

    Args:
        **kwargs: Ignore any additional arguments added in the future.

    Returns:
        A Pipeline object containing all the model training nodes.
    """
    return pipeline(
        [
            node(
                func=tune_model_hyperparameters,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:n_trials",
                    "params:random_state",
                ],
                outputs="best_hyperparameters",
                name="tune_model_hyperparameters_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "best_hyperparameters",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
        ]
    ) 