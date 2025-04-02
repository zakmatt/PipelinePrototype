"""Data processing pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from insurance_prediction.pipelines.data_processing.nodes import (
    download_data,
    load_data,
    preprocess_data,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline.

    Args:
        **kwargs: Ignore any additional arguments added in the future.

    Returns:
        A Pipeline object containing all the data processing nodes.
    """
    return pipeline(
        [
            node(
                func=download_data,
                inputs=[
                    "params:data_url",
                    "params:output_directory",
                    "params:file_name",
                ],
                outputs="raw_data_path",
                name="download_data_node",
            ),
            node(
                func=load_data,
                inputs="raw_data_path",
                outputs="raw_data",
                name="load_data_node",
            ),
            node(
                func=preprocess_data,
                inputs=["raw_data", "params:categorical_columns"],
                outputs="preprocessed_data",
                name="preprocess_data_node",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_data", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
        ]
    )
