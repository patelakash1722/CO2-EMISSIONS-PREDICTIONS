import unittest
import pandas as pd
from app import df, df_natural, df_new_model
from app import create_visualizations, create_model

class TestCO2EmissionsApp(unittest.TestCase):

    def test_data_loading(self):
        """Test if the dataset loads correctly."""
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "The dataset should not be empty.")

    def test_remove_natural_gas(self):
        """Test if rows with natural gas as fuel type are removed."""
        self.assertNotIn('Natural Gas', df_natural['Fuel Type'].unique())

    def test_outliers_removal(self):
        """Test if outliers are removed correctly."""
        original_count = df_natural.shape[0]
        new_count = df_new_model.shape[0]
        self.assertLess(new_count, original_count, "Outliers should be removed.")

    def test_visualizations_creation(self):
        """Test if visualizations are created without errors."""
        try:
            create_visualizations(df)
        except Exception as e:
            self.fail(f"Visualization creation failed with exception: {e}")

    def test_model_creation(self):
        """Test if the model is created and performs predictions without errors."""
        try:
            model, X_test, y_test = create_model(df_new_model)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(y_test), "Number of predictions should match number of test samples.")
        except Exception as e:
            self.fail(f"Model creation failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()

