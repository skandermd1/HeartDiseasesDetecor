# Heart Disease Prediction

A machine learning project for predicting whether a patient is likely to have heart disease based on clinical attributes. The project includes a training pipeline, MLflow experiment tracking, and a FastAPI web service for serving the trained model.

## Project overview

- Dataset: heart disease data in `data/heart_disease.csv`
- Training: compare multiple classifiers and log results to MLflow
- Model serving: FastAPI endpoint for predictions
- Validation: unit tests for training logic
- Deployment: Docker support and GitHub Actions CI

## Repository structure

```text
.
├── app/
│   └── main.py            # FastAPI application
├── data/
│   ├── EDA.ipynb          # Exploratory analysis notebook
│   └── heart_disease.csv  # Input dataset
├── src/
│   ├── DataIngestor.py    # Data loading logic
│   ├── DataProcessor.py   # Missing value handling
│   ├── DataSplitter.py    # Train/test splitting
│   └── ModelBuilder.py    # Model training and MLflow logging
├── tests/
│   └── test_model_builder.py
├── .github/
│   └── workflows/
│       └── ci.yml         # CI pipeline
├── Dockerfile
├── requirements.txt
├── mlruns/                # MLflow tracking artifact store
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the models

Run the training workflow from the project root:

```bash
python -c "from src.ModelBuilder import LogisticRegresorTraining; from src.DataSplitter import DataSplitter; from src.DataIngestor import DataIngestor; import pandas as pd; df = DataIngestor().read_csv('data/heart_disease.csv'); splitter = DataSplitter(); X_train, X_test, y_train, y_test = splitter.split(df); trainer = LogisticRegresorTraining(); trainer.compare_models(X_train, X_test, y_train, y_test)"
```

You can also train a specific model directly if needed.

## Run the API locally

Start the FastAPI app:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- http://localhost:8000/health
- http://localhost:8000/docs

## Example prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

Example response:

```json
{
  "prediction": 1,
  "label": "Heart disease likely",
  "probability": 0.82
}
```

## Run tests

```bash
python -m unittest tests.test_model_builder
```

## Docker

Build the image:

```bash
docker build -t heart .
```

Run the container:

```bash
docker run -d --name heart -p 8000:8000 heart
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

## MLflow

The app loads the best model from the `mlruns` artifact store. You can inspect experiments with:

```bash
mlflow ui
```

Then open:

- http://localhost:5000

## Notes

- The project uses MLflow file-based tracking.
- Missing values are imputed with median values before training and inference.
- The API loads the strongest model from the latest experiment run automatically.
- The project is structured to support extension with additional model comparisons or deployment pipelines.
