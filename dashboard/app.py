import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("models/xgb.pkl", "rb") as f:
    model = pickle.load(f)

st.title("CPU Usage Prediction Dashboard")
st.write("Predict CPU usage based on resource requests and container properties.")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

cpu_request = st.sidebar.number_input("CPU Request (in cores)", min_value=0.0, value=0.5)
mem_request = st.sidebar.number_input("Memory Request (in MB)", min_value=0.0, value=512.0)
cpu_limit = st.sidebar.number_input("CPU Limit (in cores)", min_value=0.0, value=1.0)
mem_limit = st.sidebar.number_input("Memory Limit (in MB)", min_value=0.0, value=1024.0)
runtime_minutes = st.sidebar.number_input("Runtime (minutes)", min_value=0.0, value=30.0)
controller_kind = st.sidebar.selectbox("Controller Kind", ["Deployment", "StatefulSet", "Job", "DaemonSet"])

# Encode controller kind
mapping = {
    "Deployment": 0,
    "StatefulSet": 1,
    "Job": 2,
    "DaemonSet": 3
}

controller_value = mapping[controller_kind]

# Prepare features for prediction
features = np.array([[cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_value]])

# Predict
if st.sidebar.button("Predict CPU Usage"):
    prediction = model.predict(features)[0]
    st.subheader("Predicted CPU Usage")
    st.metric(label="CPU Usage (cores)", value=round(prediction, 4))

# Visualization Section
st.header("Data Visualizations")

uploaded_data = st.file_uploader("Upload processed.csv for analysis (optional)", type=["csv"])

if uploaded_data:
    df = pd.read_csv(uploaded_data)
    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("CPU Usage Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['cpu_usage'], kde=True, ax=ax2)
    st.pyplot(fig2)

st.info("Model loaded from 'models/xgb.pkl'. Data version controlled with DVC.")

st.header("Feature Importance")
importance = model.feature_importances_
feat_names = ["cpu_request", "mem_request", "cpu_limit", "mem_limit", "runtime_minutes", "controller_kind"]

fig3, ax3 = plt.subplots()
sns.barplot(x=importance, y=feat_names, ax=ax3)
st.pyplot(fig3)

