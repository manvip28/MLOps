import mlflow

DAGSHUB_OWNER = "manvip28"
DAGSHUB_REPO = "cpu-usage-mlops"
TRACKING_URI = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"

mlflow.set_tracking_uri(TRACKING_URI)

# Run IDs
RUN_IDS = {
    'XGBoost': '1584ae5a899e4e14b0f699322958f367',
    'Random Forest': '157b4c205abe4cd985bf632db59ab949',
    'Linear Regression': '5b01152abc904e4f93a555eb0060eaae'
}

client = mlflow.MlflowClient()

for model_name, run_id in RUN_IDS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")
    
    try:
        run = client.get_run(run_id)
        
        print("\nAll Metrics:")
        for key, value in run.data.metrics.items():
            print(f"  {key}: {value}")
        
        print("\nAll Params:")
        for key, value in run.data.params.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"ERROR: {e}")