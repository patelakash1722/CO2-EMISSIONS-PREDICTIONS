import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # Import train_test_split

class TestCO2EmissionsModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test data and model for testing."""
        # Generate synthetic data for testing
        np.random.seed(42)
        data = {
            'Engine Size(L)': np.random.uniform(1.0, 5.0, 100),
            'Cylinders': np.random.randint(2, 16, 100),
            'Fuel Consumption Comb (L/100 km)': np.random.uniform(5.0, 15.0, 100),
            'CO2 Emissions(g/km)': np.random.uniform(50, 400, 100)
        }
        cls.df = pd.DataFrame(data)

        # Prepare the feature and target variables
        cls.X = cls.df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
        cls.y = cls.df['CO2 Emissions(g/km)']

    def test_prediction_accuracy(self):
        """Test the accuracy of the model's predictions."""
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor().fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Check if Mean Squared Error is within an acceptable range
        self.assertLess(mse, 5000, "Mean Squared Error is too high.")
        # Check if R^2 Score is above a threshold (e.g., 0.5 for this example)
        self.assertGreater(r2, 0.5, "R^2 Score is too low.")

    def test_single_prediction(self):
        """Test single prediction for given input data."""
        model = RandomForestRegressor().fit(self.X, self.y)
        test_input = np.array([[2.5, 4, 10.0]])
        prediction = model.predict(test_input)
        self.assertEqual(len(prediction), 1, "Prediction should return one value.")
        self.assertTrue(50 <= prediction[0] <= 400, "Predicted CO2 emission should be within the expected range.")

if __name__ == '__main__':
    unittest.main()
