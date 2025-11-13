import os
import sys
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    print("Install xgboost: pip install xgboost")
    sys.exit(1)

# -------------------------------
# MLflow Remote Setup (DagsHub)
# -------------------------------
mlflow.set_tracking_uri("https://dagshub.com/manvip28/cpu-usage-mlops.mlflow")
mlflow.set_experiment("cpu-usage-experiment")


def train_and_log(X_train, X_test, y_train, y_test, model, model_name, feature_name_type):
    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # ------------------------------
    # Metrics block (FIXED)
    # ------------------------------
    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
    }

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_version", os.getenv("DVC_DATA_VERSION", "unknown"))

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # -------- Residual Plot (saved temporarily)
        fig_path = "residuals_temp.png"
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, test_preds, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Residuals ({model_name})")
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)

        # -------- Feature importance / coefficient plot
        try:
            if model_name == "linear":
                importance = model.coef_
            else:
                importance = model.feature_importances_

            fig_path = "feature_importance_temp.png"
            plt.figure(figsize=(8, 6))
            plt.barh(X_train.columns, importance)
            plt.title(f"Feature Importance ({model_name})")
            plt.savefig(fig_path)
            mlflow.log_artifact(fig_path)
            os.remove(fig_path)
        except:
            pass

        # -------- Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(1),
            pip_requirements=[
                "scikit-learn",
                "numpy",
                "pandas",
                "xgboost",
            ]
        )

        print(f"✔ Logged {model_name} to MLflow (DagsHub)")


def train_all(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("cpu_usage", axis=1)
    y = df["cpu_usage"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "linear": (LinearRegression(), "Coefficient"),
        "rf": (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), "Feature Importance"),
        "xgb": (XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        ), "Feature Importance")
    }

    for model_name, (model, feature_type) in models.items():
        train_and_log(X_train, X_test, y_train, y_test,
                      model, model_name, feature_type)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path>")
        sys.exit(1)

    train_all(sys.argv[1])
