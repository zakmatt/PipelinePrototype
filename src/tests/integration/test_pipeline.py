"""Integration tests for the project pipelines."""

from kedro.pipeline import Pipeline

from insurance_prediction.pipelines import data_processing as dp
from insurance_prediction.pipelines import model_training as mt
from insurance_prediction.pipelines import model_evaluation as me


class TestPipelines:
    """Test class for the project pipelines."""

    def test_data_processing_pipeline_creation(self):
        """Test that the data processing pipeline can be created."""
        pipeline = dp.create_pipeline()
        assert isinstance(pipeline, Pipeline)
        
        # Check that the pipeline has the expected nodes
        expected_nodes = [
            "download_data_node", 
            "load_data_node", 
            "preprocess_data_node", 
            "split_data_node"
        ]
        node_names = [node.name for node in pipeline.nodes]
        
        for expected_node in expected_nodes:
            assert expected_node in node_names

    def test_model_training_pipeline_creation(self):
        """Test that the model training pipeline can be created."""
        pipeline = mt.create_pipeline()
        assert isinstance(pipeline, Pipeline)
        
        # Check that the pipeline has the expected nodes
        expected_nodes = [
            "tune_model_hyperparameters_node",
            "train_model_node"
        ]
        node_names = [node.name for node in pipeline.nodes]
        
        for expected_node in expected_nodes:
            assert expected_node in node_names

    def test_model_evaluation_pipeline_creation(self):
        """Test that the model evaluation pipeline can be created."""
        pipeline = me.create_pipeline()
        assert isinstance(pipeline, Pipeline)
        
        # Check that the pipeline has the expected nodes
        expected_nodes = [
            "evaluate_model_node",
            "plot_confusion_matrix_node"
        ]
        node_names = [node.name for node in pipeline.nodes]
        
        for expected_node in expected_nodes:
            assert expected_node in node_names 