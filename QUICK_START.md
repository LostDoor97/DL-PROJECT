# 🚀 QUICK REFERENCE CARD

## One-Command Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything
python run_pipeline.py

# 3. View results
mlflow ui
```

## Essential Commands

| Task | Command |
|------|---------|
| **Setup Check** | `python setup_check.py` |
| **Train Single Ticker** | `python train.py` |
| **Train Multiple (3)** | `python run_pipeline.py` |
| **Generate Predictions** | `python predict.py` |
| **Start API** | `python app.py` |
| **Launch Dashboard** | `streamlit run streamlit_app.py` |
| **View MLflow** | `mlflow ui --port 5000` |
| **Drift Reports** | `python drift_monitor.py` |
| **Docker Up** | `docker-compose up -d` |
| **Docker Down** | `docker-compose down` |

## API Endpoints

```bash
# Health check
curl http://localhost:8000/

# List tickers
curl http://localhost:8000/tickers

# Get predictions
curl "http://localhost:8000/predict?ticker=MCC.V&steps=30&smoothing=true"

# Model info
curl http://localhost:8000/models/MCC.V
```

## File Structure

```
DL/
├── config.py              # Settings
├── data_pipeline.py       # Data processing
├── model.py               # LSTM architecture  
├── train.py               # Training
├── predict.py             # Predictions
├── baseline_arima.py      # ARIMA baseline
├── app.py                 # FastAPI
├── streamlit_app.py       # Dashboard
├── drift_monitor.py       # Monitoring
├── run_pipeline.py        # Full pipeline
├── setup_check.py         # Verification
├── requirements.txt       # Dependencies
├── Dockerfile             # Container
├── docker-compose.yml     # Multi-service
└── README.md             # Full docs
```

## URLs After Deployment

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

## Key Configuration (config.py)

```python
TICKERS = ['MCC.V', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
WINDOW_SIZE = 2500
UNROLLINGS = 50
BATCH_SIZE = 32
EPOCHS = 50
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.5
```

## Quick Customization

**Add ticker:**
```python
# In config.py
TICKERS = ['MCC.V', 'NEW_TICKER', ...]
```

**Change epochs:**
```python
# In config.py
EPOCHS = 100
```

**Adjust prediction steps:**
```python
# In config.py
PREDICTION_STEPS = 60
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No module found | `pip install -r requirements.txt` |
| Port in use | Change port in `config.py` |
| No model found | Run `python train.py` first |
| CUDA errors | Install `tensorflow-cpu` instead |
| Memory error | Reduce `BATCH_SIZE` in `config.py` |

## Quick Test Workflow

```bash
# 1. Verify setup
python setup_check.py

# 2. Train one model (fast test)
python -c "from train import MLflowTrainer; MLflowTrainer('AAPL').train_model(epochs=5)"

# 3. Check it worked
python -c "from predict import find_latest_model; print(find_latest_model('AAPL'))"

# 4. Get prediction
python predict.py

# 5. View in MLflow
mlflow ui
```

## Docker Quick Reference

```bash
# Build
docker build -t stock-prediction .

# Run API
docker run -p 8000:8000 stock-prediction

# Full stack
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Remove volumes
docker-compose down -v
```

## Data Flow

```
yfinance → Raw CSV → Mid-Price → Normalized → Sequences → LSTM → Predictions → EMA → Results
```

## Model Architecture

```
Input (50, 1)
    ↓
LSTM(64, return_sequences=True)
    ↓
LSTM(64)
    ↓
Dense(32)
    ↓
Dropout(0.5)
    ↓
Dense(1)
    ↓
Output (normalized price)
```

## Metrics Tracked

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- **Direction Accuracy** (% correct up/down predictions)

## Success Criteria

✅ Direction Accuracy > 60%  
✅ MLflow Runs > 10  
✅ CI/CD Pipeline Passes  
✅ Drift Reports Generated  

## Next Steps

1. Run `setup_check.py`
2. Run `run_pipeline.py` 
3. Open MLflow UI
4. Launch Streamlit dashboard
5. Explore predictions!

## Getting Help

- See `README.md` for full documentation
- See `PROJECT_SUMMARY.md` for implementation details
- See `VENV_SETUP.md` for environment issues
