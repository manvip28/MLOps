import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from PIL import Image
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Public DagsHub MLflow
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

# Page config
st.set_page_config(
    page_title="CPU Usage MLOps Dashboard",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------- FETCH RUN METRICS --------
@st.cache_data(ttl=600)
def fetch_run_metrics(run_id, model_name):
    """Fetch metrics for a specific run"""
    try:
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        
        # Get all metrics
        all_metrics = run.data.metrics
        
        # Use test metrics (the ones visible in MLflow UI)
        rmse = all_metrics.get('test_rmse', 0)
        mae = all_metrics.get('test_mae', 0)
        r2 = all_metrics.get('test_r2', 0)
        
        # Calculate MSE from RMSE if not available
        mse = all_metrics.get('test_mse', rmse ** 2 if rmse > 0 else 0)
        
        # MAPE if available
        mape = all_metrics.get('test_mape', all_metrics.get('mape', 0))
        
        return {
            'model_name': model_name,
            'run_id': run_id,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse,
            'mape': mape,
            'status': run.info.status,
            'start_time': run.info.start_time
        }
    except Exception as e:
        st.error(f"Error fetching {model_name}: {e}")
        st.exception(e)
        return None

# -------- LOAD MODEL --------
@st.cache_resource
def load_model(run_id):
    try:
        model_uri = f"runs:/{run_id}/model"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# -------- LOAD ARTIFACTS --------
@st.cache_data
def load_artifact_image(run_id, artifact_path):
    """Download artifact from MLflow and return as PIL Image"""
    try:
        client = mlflow.MlflowClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
            with Image.open(local_path) as img:
                img_copy = img.copy()
            return img_copy
    except Exception as e:
        st.warning(f"Could not load {artifact_path} for run {run_id[:8]}: {e}")
        return None

# -------- FETCH ALL MODELS --------
with st.spinner("🔄 Fetching model metrics from MLflow..."):
    models_data = []
    for model_name, run_id in RUN_IDS.items():
        metrics = fetch_run_metrics(run_id, model_name)
        if metrics:
            models_data.append(metrics)

if not models_data:
    st.error("❌ Could not fetch any model data")
    st.stop()

df_models = pd.DataFrame(models_data)

# Find best model
best_idx = df_models['r2'].idxmax()
best_model = df_models.iloc[best_idx]

# -------- HEADER --------
st.title("🖥️ CPU Usage Prediction - MLOps Dashboard")
st.markdown("### 📊 Comprehensive Model Comparison & Prediction System")

# -------- TABS --------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Model Comparison", "🔮 Prediction", "🖼️ Visualizations"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.markdown("## 🎯 Key Performance Indicators")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏆 Best Model",
            value=best_model['model_name'],
            delta=f"R² = {best_model['r2']:.4f}"
        )
    
    with col2:
        st.metric(
            label="📉 Best RMSE",
            value=f"{best_model['rmse']:.4f}",
            delta="Lowest error"
        )
    
    with col3:
        st.metric(
            label="📊 Total Models",
            value=len(df_models),
            delta="Trained & Compared"
        )
    
    with col4:
        avg_r2 = df_models['r2'].mean()
        st.metric(
            label="📈 Avg R² Score",
            value=f"{avg_r2:.4f}",
            delta=f"Across {len(df_models)} models"
        )
    
    st.markdown("---")
    
    # Performance comparison chart
    st.markdown("## 📊 Model Performance Overview")
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("RMSE Comparison (Lower is Better)", "R² Score Comparison (Higher is Better)", "MAE Comparison (Lower is Better)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    # RMSE chart
    fig.add_trace(
        go.Bar(
            x=df_models['model_name'],
            y=df_models['rmse'],
            name='RMSE',
            marker_color=colors,
            text=df_models['rmse'].round(4),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # R2 chart
    fig.add_trace(
        go.Bar(
            x=df_models['model_name'],
            y=df_models['r2'],
            name='R² Score',
            marker_color=colors,
            text=df_models['r2'].round(4),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # MAE chart
    fig.add_trace(
        go.Bar(
            x=df_models['model_name'],
            y=df_models['mae'],
            name='MAE',
            marker_color=colors,
            text=df_models['mae'].round(4),
            textposition='outside'
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("## 📋 Detailed Metrics Table")
    
    display_df = df_models[['model_name', 'rmse', 'mae', 'r2', 'mse']].copy()
    display_df.columns = ['Model', 'RMSE ↓', 'MAE ↓', 'R² Score ↑', 'MSE ↓']
    
    st.dataframe(
        display_df.style.highlight_min(subset=['RMSE ↓', 'MAE ↓', 'MSE ↓'], color='lightgreen')
                       .highlight_max(subset=['R² Score ↑'], color='lightgreen')
                       .format({
                           'RMSE ↓': '{:.4f}',
                           'MAE ↓': '{:.4f}',
                           'R² Score ↑': '{:.4f}',
                           'MSE ↓': '{:.4f}'
                       }),
        height=200
    )
    
    # Model ranking
    st.markdown("## 🏅 Model Ranking")
    df_sorted = df_models.sort_values('r2', ascending=False).reset_index(drop=True)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (i, model) in enumerate(df_sorted.iterrows()):
        col = [col1, col2, col3][idx]
        with col:
            medal = ["🥇", "🥈", "🥉"][idx]
            st.markdown(f"""
            ### {medal} Rank {idx + 1}: {model['model_name']}
            - **R² Score:** {model['r2']:.4f}
            - **RMSE:** {model['rmse']:.4f}
            - **MAE:** {model['mae']:.4f}
            """)

# ==================== TAB 2: MODEL COMPARISON ====================
with tab2:
    st.markdown("## 📊 Comprehensive Model Comparison")
    
    # Radar chart
    st.markdown("### 🎯 Multi-Metric Radar Chart")
    
    fig_radar = go.Figure()
    
    for idx, row in df_models.iterrows():
        # Normalize metrics (invert RMSE and MAE so higher is better)
        rmse_norm = 1 - (row['rmse'] / df_models['rmse'].max())
        mae_norm = 1 - (row['mae'] / df_models['mae'].max())
        r2_norm = row['r2']
        mse_norm = 1 - (row['mse'] / df_models['mse'].max())
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[rmse_norm, mae_norm, r2_norm, mse_norm],
            theta=['RMSE Score', 'MAE Score', 'R² Score', 'MSE Score'],
            fill='toself',
            name=row['model_name'],
            line=dict(color=colors[idx])
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                ticks='outside'
            )
        ),
        showlegend=True,
        height=500,
        title="Performance Radar (Higher is Better for all metrics)"
    )
    
    st.plotly_chart(fig_radar)
    
    st.markdown("---")
    
    # Grouped bar chart
    st.markdown("### 📊 All Metrics Side-by-Side")
    
    metrics_data = []
    for _, row in df_models.iterrows():
        metrics_data.extend([
            {'Model': row['model_name'], 'Metric': 'RMSE', 'Value': row['rmse']},
            {'Model': row['model_name'], 'Metric': 'MAE', 'Value': row['mae']},
            {'Model': row['model_name'], 'Metric': 'R²', 'Value': row['r2']},
            {'Model': row['model_name'], 'Metric': 'MSE', 'Value': row['mse']},
        ])
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig_grouped = px.bar(
        df_metrics,
        x='Metric',
        y='Value',
        color='Model',
        barmode='group',
        title="All Performance Metrics Comparison",
        text='Value'
    )
    fig_grouped.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_grouped.update_layout(height=500)
    st.plotly_chart(fig_grouped, use_container_width=True)
    
    # Heatmap
    st.markdown("### 🔥 Performance Heatmap")
    
    heatmap_data = df_models[['model_name', 'rmse', 'mae', 'r2', 'mse']].set_index('model_name')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['RMSE', 'MAE', 'R²', 'MSE'],
        y=heatmap_data.index,
        colorscale='RdYlGn_r',
        text=heatmap_data.values.round(4),
        texttemplate='%{text}',
        textfont={"size": 14},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Model Performance Heatmap",
        height=300
    )
    st.plotly_chart(fig_heatmap)

# ==================== TAB 3: PREDICTION ====================
with tab3:
    st.markdown("## 🔮 Make Predictions with Selected Model")
    
    # Model selector
    selected_model_name = st.selectbox(
        "Select Model for Prediction",
        options=df_models['model_name'].tolist(),
        index=best_idx
    )
    
    selected_run = df_models[df_models['model_name'] == selected_model_name].iloc[0]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"""
        **Selected Model:** {selected_run['model_name']}  
        **Run ID:** `{selected_run['run_id']}`  
        **RMSE:** {selected_run['rmse']:.4f} | **MAE:** {selected_run['mae']:.4f} | **R²:** {selected_run['r2']:.4f}
        """)
    
    with col2:
        if selected_run['model_name'] == best_model['model_name']:
            st.success("🏆 Best Model!")
        else:
            rank = df_models.sort_values('r2', ascending=False).index.get_loc(
                df_models[df_models['model_name'] == selected_model_name].index[0]
            ) + 1
            st.warning(f"Rank: #{rank}")
    
    # Load model
    with st.spinner(f"Loading {selected_model_name} model..."):
        model = load_model(selected_run['run_id'])
    
    if model:
        st.markdown("### 📝 Enter Input Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_request = st.number_input("CPU Request", min_value=0.0, value=0.1, step=0.01, help="CPU cores requested")
            mem_request = st.number_input("Memory Request (MB)", min_value=0.0, value=128.0, step=10.0)
        
        with col2:
            cpu_limit = st.number_input("CPU Limit", min_value=0.0, value=1.0, step=0.1, help="Maximum CPU cores")
            mem_limit = st.number_input("Memory Limit (MB)", min_value=0.0, value=512.0, step=10.0)
        
        with col3:
            runtime_minutes = st.number_input("Runtime (minutes)", min_value=0.0, value=10.0, step=1.0)
            controller_kind = st.selectbox(
                "Controller Kind",
                options=["Job", "ReplicaSet", "ReplicationController", "StatefulSet"]
            )
        
        # Prediction button
        if st.button("🚀 Predict CPU Usage", type="primary"):
            # Create input data with one-hot encoding
            input_data = pd.DataFrame([{
                "cpu_request": cpu_request,
                "mem_request": mem_request,
                "cpu_limit": cpu_limit,
                "mem_limit": mem_limit,
                "runtime_minutes": runtime_minutes,
                "controller_kind_Job": 1 if controller_kind == "Job" else 0,
                "controller_kind_ReplicaSet": 1 if controller_kind == "ReplicaSet" else 0,
                "controller_kind_ReplicationController": 1 if controller_kind == "ReplicationController" else 0,
                "controller_kind_StatefulSet": 1 if controller_kind == "StatefulSet" else 0
            }])
            
            try:
                with st.spinner("Predicting..."):
                    pred = model.predict(input_data)[0]
                
                st.markdown("---")
                st.markdown("## 🎯 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Predicted CPU Usage", value=f"{pred:.4f} cores")
                
                with col2:
                    confidence = (selected_run['r2'] * 100)
                    st.metric(label="Model Confidence (R²)", value=f"{confidence:.2f}%")
                
                with col3:
                    error_margin = selected_run['rmse']
                    st.metric(label="Expected Error (±)", value=f"{error_margin:.4f}")
                
                # Gauge visualization
                st.markdown("### 📊 Prediction Visualization")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU Usage (cores)", 'font': {'size': 24}},
                    delta={'reference': cpu_request, 'suffix': ' vs requested'},
                    gauge={
                        'axis': {'range': [None, max(1.5, pred + 0.2)]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgreen"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1.5], 'color': "orange"},
                            {'range': [1.5, 2], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': cpu_limit
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig)
                
                # Comparison with all models
                st.markdown("### 📊 How Would Other Models Predict?")
                
                all_predictions = []
                for _, model_row in df_models.iterrows():
                    try:
                        temp_model = load_model(model_row['run_id'])
                        if temp_model:
                            temp_pred = temp_model.predict(input_data)[0]
                            all_predictions.append({
                                'Model': model_row['model_name'],
                                'Prediction': temp_pred,
                                'R² Score': model_row['r2']
                            })
                    except:
                        pass
                
                if all_predictions:
                    df_predictions = pd.DataFrame(all_predictions)
                    
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        x=df_predictions['Model'],
                        y=df_predictions['Prediction'],
                        text=df_predictions['Prediction'].round(4),
                        textposition='outside',
                        marker_color=colors[:len(df_predictions)]
                    ))
                    fig_compare.update_layout(
                        title="Prediction Comparison Across All Models",
                        yaxis_title="CPU Usage (cores)",
                        height=400
                    )
                    st.plotly_chart(fig_compare)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)
    else:
        st.error("Failed to load model")

# ==================== TAB 4: VISUALIZATIONS ====================
with tab4:
    st.markdown("## 🖼️ Model Artifacts & Visualizations")
    st.markdown("### Compare residual plots and feature importance across all models")
    
    # Residual Plots
    st.markdown("## 📊 Residual Plots - All Models")
    st.markdown("*Lower and more randomly distributed residuals indicate better model fit*")
    
    cols = st.columns(3)
    
    for idx, (_, model) in enumerate(df_models.iterrows()):
        with cols[idx]:
            st.markdown(f"### {model['model_name']}")
            st.caption(f"R² = {model['r2']:.4f} | RMSE = {model['rmse']:.4f}")
            
            with st.spinner(f"Loading residuals for {model['model_name']}..."):
                residual_img = load_artifact_image(model['run_id'], "residuals_temp.png")
            
            if residual_img:
                st.image(residual_img, width=400)
            else:
                st.error("❌ Residual plot not available")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("## 📈 Feature Importance - All Models")
    st.markdown("*Shows which features contribute most to predictions*")
    
    cols = st.columns(3)
    
    for idx, (_, model) in enumerate(df_models.iterrows()):
        with cols[idx]:
            st.markdown(f"### {model['model_name']}")
            st.caption(f"MAE = {model['mae']:.4f}")
            
            with st.spinner(f"Loading feature importance for {model['model_name']}..."):
                feature_img = load_artifact_image(model['run_id'], "feature_importance_temp.png")
            
            if feature_img:
                st.image(feature_img, width=400)
            else:
                st.error("❌ Feature importance plot not available")
    
    st.markdown("---")
    
    # Model insights
    st.markdown("## 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Best Performing Model")
        st.success(f"""
        **{best_model['model_name']}** shows the best performance with:
        - Highest R² Score: {best_model['r2']:.4f}
        - Lowest RMSE: {best_model['rmse']:.4f}
        - Lowest MAE: {best_model['mae']:.4f}
        
        This model explains {best_model['r2']*100:.2f}% of the variance in CPU usage.
        """)
    
    with col2:
        st.markdown("### 📉 Performance Gap")
        worst_model = df_models.loc[df_models['r2'].idxmin()]
        gap = (best_model['r2'] - worst_model['r2']) / worst_model['r2'] * 100
        
        st.info(f"""
        Performance difference between best and worst model:
        - **Best:** {best_model['model_name']} (R² = {best_model['r2']:.4f})
        - **Worst:** {worst_model['model_name']} (R² = {worst_model['r2']:.4f})
        - **Gap:** {gap:.2f}% improvement
        """)

# -------- SIDEBAR --------
with st.sidebar:
    st.markdown("## ℹ️ Dashboard Info")
    st.markdown(f"""
    - **MLflow URI:** [View]({TRACKING_URI})
    - **Total Models:** {len(df_models)}
    - **Best Model:** {best_model['model_name']}
    - **Best R²:** {best_model['r2']:.4f}
    - **Best RMSE:** {best_model['rmse']:.4f}
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Model Statistics")
    
    for _, model in df_models.iterrows():
        with st.expander(f"📁 {model['model_name']}"):
            st.markdown(f"""
            **Run ID:** `{model['run_id'][:16]}...`
            
            **Metrics:**
            - RMSE: {model['rmse']:.4f}
            - MAE: {model['mae']:.4f}
            - R²: {model['r2']:.4f}
            - MSE: {model['mse']:.4f}
            """)
    
    st.markdown("---")
    st.markdown("## 🔄 Data Freshness")
    st.info("Data cached for 10 minutes")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎓 About")
    st.caption("MLOps Dashboard for CPU Usage Prediction comparing XGBoost, Random Forest, and Linear Regression models.")