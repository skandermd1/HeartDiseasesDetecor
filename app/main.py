import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.DataProcessor import Processor

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class PatientInput(BaseModel):
    age: float = Field(..., ge=0)
    sex: float = Field(..., ge=0, le=1)
    cp: float = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: float = Field(..., ge=0, le=3)
    ca: float | None = Field(default=None, ge=0, le=4)
    thal: float | None = Field(default=None, ge=0, le=7)


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


def load_best_model_from_artifacts():
    tracking_dir = Path(__file__).resolve().parent.parent / "mlruns"
    mlflow.set_tracking_uri(str(tracking_dir))

    experiment = mlflow.get_experiment_by_name("heart_disease_prediction")
    if experiment is None:
        raise RuntimeError("No heart_disease_prediction experiment found in MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=10,
    )
    if runs.empty:
        raise RuntimeError("No model runs found in the MLflow artifacts.")

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    model_name = best_run.get("tags.mlflow.runName")
    if not model_name:
        model_name = "logistic_regression"

    model_uri = f"runs:/{best_run_id}/{model_name}_model"
    return mlflow.sklearn.load_model(model_uri)


try:
    MODEL = load_best_model_from_artifacts()
except Exception:
    MODEL = None


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientInput):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="No model artifact found in MLflow.")

    payload = patient.model_dump()
    payload = {key: payload.get(key) for key in FEATURE_COLUMNS}
    features = pd.DataFrame([payload], columns=FEATURE_COLUMNS)

    processor = Processor()
    features = processor.HandleMissingValues(features)

    prediction = int(MODEL.predict(features)[0])
    probability = None
    if hasattr(MODEL, "predict_proba"):
        probability = float(MODEL.predict_proba(features)[0, 1])

    return {
        "prediction": prediction,
        "label": "Heart disease likely" if prediction == 1 else "No heart disease detected",
        "probability": probability,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)