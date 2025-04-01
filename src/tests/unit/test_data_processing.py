"""Unit tests for data processing nodes."""

import pandas as pd

from insurance_prediction.pipelines.data_processing.nodes import preprocess_data, split_data


class TestDataProcessing:
    """Test class for data processing nodes."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a sample dataframe for testing
        self.data = pd.DataFrame({
            'Numtppd': [0, 1, 2, 0],
            'Numtpbi': [0, 0, 1, 1],
            'Indtppd': [0, 1, 1, 0],
            'Indtpbi': [0, 0, 1, 1],
            'Age': [25, 30, 45, 50],
            'Gender': ['M', 'F', 'M', 'F'],
            'CalYear': [2020, 2020, 2021, 2021],
        })

    def test_preprocess_data(self):
        """Test the preprocess_data function."""
        # Define categorical columns
        categorical_columns = ['Gender', 'CalYear']
        
        # Call the function
        result = preprocess_data(self.data, categorical_columns)
        
        # Check if target column is created
        assert 'target' in result.columns
        
        # Check if dropped columns are not in result
        for col in ['Numtppd', 'Numtpbi', 'Indtppd', 'Indtpbi']:
            assert col not in result.columns
        
        # Check one-hot encoding
        assert 'Gender_M' in result.columns
        assert 'Gender_F' in result.columns
        assert 'CalYear_2020' in result.columns
        assert 'CalYear_2021' in result.columns
        
        # Check target values
        assert result['target'].tolist() == [0, 1, 1, 0]

    def test_split_data(self):
        """Test the split_data function."""
        # Prepare test data with target column
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Call the function
        X_train, X_test, y_train, y_test = split_data(data, test_size=0.4, random_state=42)
        
        # Check shapes
        assert X_train.shape[0] == 3  # 60% of 5 rows
        assert X_test.shape[0] == 2   # 40% of 5 rows
        assert y_train.shape[0] == 3
        assert y_test.shape[0] == 2
        
        # Check that target column is not in X
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
        
        # Check that other columns are in X
        assert 'feature1' in X_train.columns
        assert 'feature2' in X_train.columns 