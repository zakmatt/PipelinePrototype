"""Model evaluation pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from insurance_prediction.pipelines.model_evaluation.nodes import (
    evaluate_model,
    plot_confusion_matrix,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model evaluation pipeline.

    Args:
        **kwargs: Ignore any additional arguments added in the future.

    Returns:
        A Pipeline object containing all the model evaluation nodes.
    """
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=plot_confusion_matrix,
                inputs=["trained_model", "X_test", "y_test", "params:output_directory"],
                outputs=None,
                name="plot_confusion_matrix_node",
            ),
        ]
    ) 