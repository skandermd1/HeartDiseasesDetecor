import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.DataProcessor import Processor

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class LogisticRegresorTraining:
    def _make_model(self, model_name, params):
        if model_name == "logistic_regression":
            return LogisticRegression(**params)
        if model_name == "decision_tree":
            return DecisionTreeClassifier(**params)
        if model_name == "random_forest":
            return RandomForestClassifier(**params)
        if model_name == "knn":
            return KNeighborsClassifier(**params)
        raise ValueError(f"Unknown model: {model_name}")

    def _compute_metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

    def train(self, X_train, X_test, y_train, y_test, model_name="logistic_regression"):
        processor = Processor()
        X_train = processor.HandleMissingValues(X_train)
        X_test = processor.HandleMissingValues(X_test)

        tracking_dir = Path(__file__).resolve().parent.parent / "mlruns"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(tracking_dir))
        mlflow.set_experiment("heart_disease_prediction")

        default_params = {
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42,
                "solver": "lbfgs",
            },
            "decision_tree": {
                "random_state": 42,
                "max_depth": 5,
                "min_samples_leaf": 2,
            },
            "random_forest": {
                "n_estimators": 200,
                "random_state": 42,
                "max_depth": 6,
                "min_samples_leaf": 2,
            },
            "knn": {
                "n_neighbors": 5,
            },
        }

        params = default_params[model_name]
        model = self._make_model(model_name, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._compute_metrics(y_test, y_pred)

        trusted_types = [
            "sklearn.tree._tree.Tree",
            "sklearn.neighbors._kd_tree.KDTree",
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
        ]

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                f"{model_name}_model",
                skops_trusted_types=trusted_types,
            )

        return model, metrics

    def compare_models(self, X_train, X_test, y_train, y_test):
        models = ["logistic_regression", "decision_tree", "random_forest", "knn"]
        results = {}

        for model_name in models:
            model, metrics = self.train(X_train, X_test, y_train, y_test, model_name=model_name)
            results[model_name] = {"model": model, "metrics": metrics}

        return results

 