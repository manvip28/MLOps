import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from PIL import Image
import tempfile
import os

DAGSHUB_OWNER = "manvip28"
DAGSHUB_REPO = "cpu-usage-mlops"
TRACKING_URI = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"

mlflow.set_tracking_uri(TRACKING_URI)

BEST_RUN_ID = "1584ae5a899e4e14b0f699322958f367"

# -------- LOAD MODEL -------- 
@st.cache_resource
def load_model():
    model_uri = f"runs:/{BEST_RUN_ID}/model"
    return mlflow.sklearn.load_model(model_uri)

# -------- LOAD ARTIFACTS --------
@st.cache_data
def load_artifact_image(artifact_path):
    """Download artifact from MLflow and return as PIL Image"""
    client = mlflow.MlflowClient()
    
    # Download artifact to a temporary location
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = client.download_artifacts(BEST_RUN_ID, artifact_path, tmpdir)
        
        # Open and load the image into memory, then close the file
        with Image.open(local_path) as img:
            # Create a copy in memory so the file can be closed
            img_copy = img.copy()
        
        return img_copy

model = load_model()

# -------- UI -------- 
st.title("CPU Usage Prediction Dashboard")

cpu_request = st.number_input("CPU Request", min_value=0.0)
mem_request = st.number_input("Memory Request (MB)", min_value=0.0)
cpu_limit = st.number_input("CPU Limit", min_value=0.0)
mem_limit = st.number_input("Memory Limit (MB)", min_value=0.0)
runtime_minutes = st.number_input("Runtime (minutes)", min_value=0.0)
controller_kind = st.number_input("Controller Kind (int)", min_value=0)

input_data = pd.DataFrame([{
    "cpu_request": cpu_request,
    "mem_request": mem_request,
    "cpu_limit": cpu_limit,
    "mem_limit": mem_limit,
    "runtime_minutes": runtime_minutes,
    "controller_kind": controller_kind
}])

if st.button("Predict CPU Usage"):
    pred = model.predict(input_data)[0]
    st.success(f"Predicted CPU Usage: {pred:.3f}")

# -------- SHOW IMAGES -------- 
st.subheader("Residual Plot")
try:
    residual_img = load_artifact_image("residuals_temp.png")
    st.image(residual_img)
except Exception as e:
    st.error(f"Could not load residual plot: {e}")

st.subheader("Feature Importance")
try:
    feature_img = load_artifact_image("feature_importance_temp.png")
    st.image(feature_img)
except Exception as e:
    st.error(f"Could not load feature importance plot: {e}")