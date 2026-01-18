"""
Streamlit Dashboard for Stock Price Predictions
Visualizes actual vs predicted prices and drift reports
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from pathlib import Path
import json

from data_pipeline import StockDataPipeline
from predict import StockPredictor, find_latest_model
from drift_monitor import DriftMonitor
from config import TICKERS, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, REPORTS_DIR

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)


def load_mlflow_runs():
    """Load MLflow runs for comparison"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        return pd.DataFrame()
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


def plot_predictions(ticker, actual_data, predictions, smoothed_predictions):
    """Create interactive plot of actual vs predicted prices"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stock Price: Actual vs Predicted', 'Prediction Comparison'),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )
    
    # Main plot - Actual prices
    fig.add_trace(
        go.Scatter(
            x=list(range(len(actual_data))),
            y=actual_data,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Predictions start from end of actual data
    pred_x = list(range(len(actual_data), len(actual_data) + len(predictions)))
    
    # Raw predictions
    fig.add_trace(
        go.Scatter(
            x=pred_x,
            y=predictions,
            mode='lines',
            name='Raw Predictions',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Smoothed predictions
    fig.add_trace(
        go.Scatter(
            x=pred_x,
            y=smoothed_predictions,
            mode='lines',
            name='Smoothed Predictions (EMA)',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Second subplot - Prediction comparison
    fig.add_trace(
        go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            mode='lines+markers',
            name='Raw',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(smoothed_predictions))),
            y=smoothed_predictions,
            mode='lines+markers',
            name='Smoothed',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time Steps", row=1, col=1)
    fig.update_xaxes(title_text="Prediction Steps", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"{ticker} - Stock Price Predictions",
        hovermode='x unified'
    )
    
    return fig


def main():
    st.title("📈 Stock Price Prediction Dashboard")
    st.markdown("### LSTM-based Stock Market Predictions with MLOps")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Ticker selection
    ticker = st.sidebar.selectbox(
        "Select Ticker",
        options=TICKERS,
        index=0
    )
    
    # Prediction steps
    pred_steps = st.sidebar.slider(
        "Prediction Steps",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )
    
    # Apply smoothing
    apply_smoothing = st.sidebar.checkbox("Apply EMA Smoothing", value=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 MLflow Runs", "🔍 Drift Reports", "ℹ️ Info"])
    
    with tab1:
        st.header(f"Predictions for {ticker}")
        
        # Check if model exists
        model_path = find_latest_model(ticker)
        
        if model_path is None:
            st.error(f"❌ No trained model found for {ticker}. Please train a model first.")
            st.info("Run `python train.py` to train a model for this ticker.")
        else:
            st.success(f"✅ Model found: {Path(model_path).name}")
            
            # Generate predictions button
            if st.button("Generate Predictions", type="primary"):
                try:
                    with st.spinner("Loading data..."):
                        # Get actual data first
                        pipeline = StockDataPipeline(ticker)
                        pipeline.fetch_data()
                        data = pipeline.compute_mid_price()
                        normalized_prices = pipeline.windowed_normalize(np.array(data['Mid'].values))
                        
                        st.success(f"✓ Data loaded: {len(normalized_prices)} points")
                    
                    with st.spinner("Loading model..."):
                        # Load model and predict
                        predictor = StockPredictor(model_path, ticker)
                        st.success("✓ Model loaded")
                    
                    with st.spinner(f"Generating {pred_steps} step predictions..."):
                        # Generate predictions
                        result = predictor.predict_from_ticker(
                            steps=pred_steps,
                            apply_smoothing=apply_smoothing
                        )
                        st.success("✓ Predictions generated")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Ticker", ticker)
                    with col2:
                        st.metric("Prediction Steps", pred_steps)
                    with col3:
                        st.metric("Last Actual Value", f"{result['last_actual_value']:.4f}")
                    with col4:
                        next_pred = result['smoothed_predictions'][0]
                        change = ((next_pred - result['last_actual_value']) / result['last_actual_value']) * 100
                        st.metric("Next Step Prediction", f"{next_pred:.4f}", f"{change:+.2f}%")
                    
                    # Plot
                    st.subheader("Price Visualization")
                    try:
                        fig = plot_predictions(
                            ticker,
                            normalized_prices[-100:],  # Show last 100 actual points
                            result['raw_predictions'],
                            result['smoothed_predictions']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as plot_error:
                        st.warning(f"Could not create plot: {plot_error}")
                        st.info("Data is available below in table format.")
                    
                    # Predictions table
                    st.subheader("Prediction Values")
                    pred_df = pd.DataFrame({
                        'Step': range(1, len(result['smoothed_predictions']) + 1),
                        'Raw Prediction': result['raw_predictions'],
                        'Smoothed Prediction': result['smoothed_predictions']
                    })
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"{ticker}_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    import traceback
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())
    
    with tab2:
        st.header("MLflow Experiment Runs")
        
        try:
            runs_df = load_mlflow_runs()
            
            if runs_df.empty:
                st.warning("No MLflow runs found. Train some models first.")
            else:
                st.success(f"Found {len(runs_df)} runs")
                
                # Filter by ticker
                ticker_filter = st.selectbox(
                    "Filter by Ticker",
                    options=["All"] + TICKERS,
                    key="mlflow_filter"
                )
                
                if ticker_filter != "All":
                    runs_df = runs_df[runs_df['params.ticker'] == ticker_filter]
                
                # Display key metrics
                if not runs_df.empty:
                    st.subheader("Run Comparison")
                    
                    # Select columns to display
                    display_cols = [
                        'run_id',
                        'params.ticker',
                        'params.model_type',
                        'metrics.direction_accuracy',
                        'metrics.test_rmse',
                        'metrics.test_mse',
                        'start_time'
                    ]
                    
                    available_cols = [col for col in display_cols if col in runs_df.columns]
                    display_df = runs_df[available_cols].copy()
                    
                    # Rename columns for display
                    display_df.columns = [col.replace('params.', '').replace('metrics.', '') for col in display_df.columns]
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Best model
                    if 'direction_accuracy' in display_df.columns:
                        best_run = display_df.loc[display_df['direction_accuracy'].idxmax()]
                        st.subheader("🏆 Best Model")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ticker", best_run['ticker'])
                        with col2:
                            st.metric("Direction Accuracy", f"{best_run['direction_accuracy']:.2f}%")
                        with col3:
                            st.metric("RMSE", f"{best_run['test_rmse']:.6f}")
                
        except Exception as e:
            st.error(f"Error loading MLflow runs: {str(e)}")
    
    with tab3:
        st.header("Data & Model Drift Reports")
        
        st.info("Generate drift reports to monitor data quality and model performance over time.")
        
        if st.button("Generate Drift Report", type="primary"):
            with st.spinner(f"Generating drift report for {ticker}..."):
                try:
                    monitor = DriftMonitor(ticker)
                    result = monitor.monitor_ticker(lookback_days=30)
                    
                    st.success("✅ Drift report generated successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Reference Samples", result['reference_samples'])
                    with col2:
                        st.metric("Current Samples", result['current_samples'])
                    
                    # Display report links
                    st.subheader("Generated Reports")
                    
                    if result['drift_report']:
                        st.markdown(f"**Data Drift Report:** `{result['drift_report']}`")
                        
                    if result['performance_report']:
                        st.markdown(f"**Performance Report:** `{result['performance_report']}`")
                    
                    st.info("💡 Open the HTML files in a browser to view detailed reports.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating drift report: {str(e)}")
        
        # List existing reports
        st.subheader("Existing Reports")
        report_files = list(REPORTS_DIR.glob("*.html"))
        
        if report_files:
            report_df = pd.DataFrame({
                'Report': [f.name for f in report_files],
                'Size (KB)': [f.stat().st_size / 1024 for f in report_files],
                'Modified': [pd.Timestamp.fromtimestamp(f.stat().st_mtime) for f in report_files]
            })
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("No drift reports generated yet.")
    
    with tab4:
        st.header("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration")
            st.markdown(f"""
            - **Available Tickers:** {len(TICKERS)}
            - **MLflow Tracking URI:** `{MLFLOW_TRACKING_URI}`
            - **Experiment:** `{EXPERIMENT_NAME}`
            - **Reports Directory:** `{REPORTS_DIR}`
            """)
        
        with col2:
            st.subheader("Model Architecture")
            st.markdown("""
            - **Type:** Stacked LSTM
            - **Layer 1:** LSTM(64, return_sequences=True)
            - **Layer 2:** LSTM(64)
            - **Dense:** Dense(32)
            - **Dropout:** 0.5
            - **Output:** Dense(1)
            - **Loss:** MSE
            - **Optimizer:** Adam
            """)
        
        st.subheader("Available Tickers")
        ticker_df = pd.DataFrame({
            'Ticker': TICKERS,
            'Model Available': [find_latest_model(t) is not None for t in TICKERS]
        })
        st.dataframe(ticker_df, use_container_width=True)
        
        st.subheader("Quick Start")
        st.code("""
# Train a model
python train.py

# Generate predictions
python predict.py

# Start API server
python app.py

# Run this dashboard
streamlit run streamlit_app.py
        """, language="bash")


if __name__ == "__main__":
    main()
