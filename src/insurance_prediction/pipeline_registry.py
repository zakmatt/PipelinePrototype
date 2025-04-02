"""Project pipelines registry."""

from typing import Dict

from kedro.pipeline import Pipeline

from insurance_prediction.pipelines import data_processing as dp
from insurance_prediction.pipelines import model_evaluation as me
from insurance_prediction.pipelines import model_training as mt


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

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
        "__default__": data_processing_pipeline
        + model_training_pipeline
        + model_evaluation_pipeline,
    }
