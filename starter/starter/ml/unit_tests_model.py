import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

import pdb


# Tested functions
from model import train_model, compute_model_metrics, inference

class TestMachineLearningFunctions(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_train = np.array([0, 1, 0])
        
        # Create a mock model for testing inference
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

        # Create sample data for inference
        self.X_test = np.array([[2, 3], [4, 5]])

    def test_train_model(self):
        # Test the train_model function
        model = train_model(self.X_train, self.y_train)
        self.assertIsInstance(model, DecisionTreeClassifier)
        self.assertTrue(hasattr(model, "predict"))

    def test_compute_model_metrics(self):
        # Test the compute_model_metrics function
        y_true = np.array([0, 1, 0])
        preds = np.array([0, 1, 1])
        precision, recall, fbeta = compute_model_metrics(y_true, preds)

        # Assert expected values
        self.assertAlmostEqual(precision, 0.5, places=2)
        self.assertAlmostEqual(recall, 1.0, places=2)
        self.assertAlmostEqual(fbeta, 0.66, places=1)

    def test_inference(self):
        # Test the inference function
        preds = inference(self.model, self.X_test)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), len(self.X_test))
        # Check if predictions are binary
        self.assertTrue(np.all(np.isin(preds, [0, 1])))  


if __name__ == "__main__":
    unittest.main()
