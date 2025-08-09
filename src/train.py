# train.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri("http://localhost:5000")

def train_model():
    # Load processed data
    X_train = pd.read_csv("data/processed/x_train.csv")
    X_test = pd.read_csv("data/processed/x_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Flatten y if needed
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Start MLflow experiment
    mlflow.set_experiment("California_Housing_LinearRegression")

    with mlflow.start_run():
        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log parameters (no hyperparams for simple LinearRegression)
        mlflow.log_param("model_type", "LinearRegression")

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f" Model trained and logged to MLflow")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    train_model()
