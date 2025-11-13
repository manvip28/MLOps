import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# --- MLflow Setup ---
mlflow.set_tracking_uri("https://dagshub.com/manvip28/cpu-usage-mlops.mlflow")
mlflow.set_experiment("mlflow-test")

# --- Dummy Data ---
X, y = make_regression(n_samples=100, n_features=3, noise=0.2)

# --- MLflow Run ---
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    # Metrics
    score = model.score(X, y)
    mlflow.log_metric("train_r2", score)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Run completed. Check your DagsHub MLflow UI.")
