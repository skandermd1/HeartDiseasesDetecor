import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ModelBuilder import LogisticRegresorTraining


class TestModelBuilder(unittest.TestCase):
    def test_train_returns_metrics(self):
        df = pd.read_csv("data/heart_disease.csv")
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        trainer = LogisticRegresorTraining()
        model, metrics = trainer.train(X_train, X_test, y_train, y_test)

        self.assertIsNotNone(model)
        self.assertTrue({"accuracy", "precision", "recall", "f1_score"}.issubset(metrics.keys()))

    def test_compare_models_returns_multiple_metrics(self):
        df = pd.read_csv("data/heart_disease.csv")
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        trainer = LogisticRegresorTraining()
        results = trainer.compare_models(X_train, X_test, y_train, y_test)

        self.assertGreater(len(results), 1)
        self.assertIn("logistic_regression", results)


if __name__ == "__main__":
    unittest.main()
