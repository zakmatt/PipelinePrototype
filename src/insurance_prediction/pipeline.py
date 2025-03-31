"""Pipeline construction."""

from kedro.pipeline import Pipeline

from insurance_prediction.pipelines import data_processing as dp
from insurance_prediction.pipelines import model_training as mt
from insurance_prediction.pipelines import model_evaluation as me


def create_pipelines() -> dict[str, Pipeline]:
    """Create the project's pipeline.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    model_evaluation_pipeline = me.create_pipeline()

    return {
        "dp": data_processing_pipeline,
        "mt": model_training_pipeline,
        "me": model_evaluation_pipeline,
        "__default__": data_processing_pipeline + model_training_pipeline + model_evaluation_pipeline,
    } 