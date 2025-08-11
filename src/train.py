# train.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib

# -----------------------
# 1. Load Dataset
# -----------------------
data_path = "data/california.csv"
df = pd.read_csv(data_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------
# 2. Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 3. MLflow Setup
# -----------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("California_Housing_Models")

os.makedirs("models", exist_ok=True)

# -----------------------
# 4. Train Linear Regression
# -----------------------
with mlflow.start_run(run_name="LinearRegression") as run:
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse_lr)
    mlflow.log_metric("r2", r2_lr)

    # Log to MLflow
    mlflow.sklearn.log_model(lr_model, artifact_path="linear_regression_model")

    # Also register to MLflow Model Registry
    model_uri = f"runs:/{run.info.run_id}/linear_regression_model"
    mlflow.register_model(model_uri, "housing_model")

    # Save locally
    joblib.dump(lr_model, "models/model.joblib")

# -----------------------
# 5. Train Decision Tree (optional extra model)
# -----------------------
with mlflow.start_run(run_name="DecisionTreeRegressor"):
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred_dt = dt_model.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)

    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.log_metric("mse", mse_dt)
    mlflow.log_metric("r2", r2_dt)

    mlflow.sklearn.log_model(dt_model, artifact_path="decision_tree_model")
    joblib.dump(dt_model, "models/decision_tree.joblib")

print("✅ Models trained, logged to MLflow, and saved locally.")
