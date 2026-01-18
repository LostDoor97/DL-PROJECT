# 📈 Stock Market Prediction with LSTM and MLOps

A comprehensive stock price prediction system using stacked LSTM neural networks with complete MLOps pipeline including MLflow tracking, Docker deployment, Evidently drift monitoring, and CI/CD automation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-blue.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)

## 🎯 Project Overview

This project implements an end-to-end MLOps pipeline for stock price prediction:

- **Data Pipeline**: Fetches OHLCV data from yfinance for 5+ tickers (5 years history)
- **Model**: Stacked LSTM (64 units/layer, dropout=0.5) with windowed normalization
- **MLOps**: MLflow for experiment tracking, Docker containerization, GitHub Actions CI/CD
- **Monitoring**: Evidently for data/model drift detection
- **Deployment**: FastAPI REST API, Streamlit dashboard
- **Baseline**: ARIMA comparison for model validation

### Key Features

✅ Multi-ticker support (MCC.V, AAPL, GOOGL, MSFT, TSLA, NVDA)  
✅ Windowed normalization (2500-point windows)  
✅ EMA smoothing for predictions  
✅ Direction accuracy >60% target  
✅ 10+ tracked MLflow runs  
✅ Reproducible CI/CD pipeline  
✅ Data & model drift monitoring  
✅ Interactive visualization dashboard  

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
│  yfinance → OHLCV → Mid-Price → Windowed Norm → Sequences   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Model                                │
│  Input → LSTM(64, return_seq=True) → LSTM(64) →             │
│  Dense(32) → Dropout(0.5) → Dense(1)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Tracking                           │
│  Params (unrollings, batch_size) → Metrics (MSE, accuracy)  │
│  → Model Artifacts (.h5) → Comparison (LSTM vs ARIMA)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Deployment                                │
│  FastAPI (/predict?ticker=MCC) ← Docker ← GitHub Actions    │
│  Streamlit Dashboard (Visuals + Drift Reports)              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd DL
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train Models

Train LSTM model for a single ticker:
```bash
python train.py
```

Train multiple tickers:
```python
from train import train_multiple_tickers
results = train_multiple_tickers(['MCC.V', 'AAPL', 'GOOGL'])
```

Train baseline ARIMA for comparison:
```bash
python baseline_arima.py
```

#### 2. Generate Predictions

```bash
python predict.py
```

Or programmatically:
```python
from predict import StockPredictor, find_latest_model

model_path = find_latest_model('MCC.V')
predictor = StockPredictor(model_path, 'MCC.V')
result = predictor.predict_from_ticker(steps=30)
```

#### 3. Start API Server

```bash
python app.py
```

API endpoints:
- `GET /` - Health check
- `GET /predict?ticker=MCC.V&steps=30` - Generate predictions
- `GET /tickers` - List available tickers
- `POST /predict/batch` - Batch predictions

Test the API:
```bash
curl "http://localhost:8000/predict?ticker=MCC.V&steps=30&smoothing=true"
```

#### 4. Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

Dashboard features:
- 📊 Interactive price predictions
- 📈 MLflow run comparison
- 🔍 Drift reports
- ℹ️ System information

#### 5. Monitor Drift

```bash
python drift_monitor.py
```

Generates:
- Data drift reports (Evidently)
- Model performance reports
- HTML visualizations in `evidently_reports/`

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

### Individual Services

```bash
# Build image
docker build -t stock-prediction .

# Run API
docker run -p 8000:8000 -v $(pwd)/models:/app/models stock-prediction python app.py

# Run Dashboard
docker run -p 8501:8501 -v $(pwd)/models:/app/models stock-prediction streamlit run streamlit_app.py
```

## 📊 MLflow Experiment Tracking

View MLflow UI:
```bash
mlflow ui --port 5000
```

Access at: http://localhost:5000

Compare runs:
- Filter by ticker, model type
- Compare metrics (MSE, RMSE, direction accuracy)
- Download best model artifacts

## 🧪 Model Details

### LSTM Architecture

```python
Sequential([
    LSTM(64, return_sequences=True, input_shape=(50, 1)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length (Unrollings) | 50 |
| Batch Size | 32 |
| Epochs | 50 |
| Learning Rate | 0.001 |
| Dropout Rate | 0.5 |
| Window Size (Normalization) | 2500 |
| Train/Test Split | 80/20 |

### Data Processing

1. **Fetch**: yfinance OHLCV data (2020-present)
2. **Mid-Price**: `(High + Low) / 2`
3. **Windowed Normalization**: MinMaxScaler per 2500-point window
4. **Sequences**: Rolling windows of 50 time steps
5. **EMA Smoothing**: Exponential moving average (span=10)

## 📈 Performance Metrics

Target metrics:
- ✅ Direction Accuracy: >60%
- ✅ RMSE: Minimized through training
- ✅ MLflow Runs: >10 tracked experiments
- ✅ CI/CD: Automated testing & deployment

## 🔄 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

1. **Lint & Test**: Code quality checks
2. **Train Model**: Automated training on push
3. **Build Docker**: Container image creation
4. **Generate Reports**: Drift monitoring
5. **Deploy**: Manual deployment trigger

Trigger workflow:
```bash
git add .
git commit -m "Update model"
git push origin main
```

## 📂 Project Structure

```
DL/
├── config.py                 # Configuration settings
├── data_pipeline.py          # Data fetching & preprocessing
├── model.py                  # LSTM model architecture
├── train.py                  # Training with MLflow
├── predict.py                # Prediction & EMA smoothing
├── baseline_arima.py         # ARIMA baseline model
├── app.py                    # FastAPI application
├── streamlit_app.py          # Streamlit dashboard
├── drift_monitor.py          # Evidently drift monitoring
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Multi-container setup
├── .github/workflows/        # CI/CD pipelines
├── data/                     # Data storage
│   ├── raw/                  # Raw CSV files
│   └── processed/            # Processed NPZ files
├── models/                   # Trained models (.h5)
├── mlruns/                   # MLflow artifacts
├── logs/                     # Application logs
└── evidently_reports/        # Drift reports
```

## 🛠️ Development

### Adding New Tickers

Edit `config.py`:
```python
TICKERS = ['MCC.V', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'NEW_TICKER']
```

### Custom Model Training

```python
from train import MLflowTrainer

trainer = MLflowTrainer('AAPL')
result = trainer.train_model(
    batch_size=64,
    epochs=100,
    learning_rate=0.0005
)
```

### Extending the API

Add new endpoints in `app.py`:
```python
@app.get("/custom-endpoint")
async def custom_endpoint():
    # Your logic here
    return {"status": "success"}
```

## 📊 Evaluation Criteria

| Criterion | Target | Implementation |
|-----------|--------|----------------|
| Pipeline Reproducibility | CI/CD Pass | ✅ GitHub Actions |
| Forecast Accuracy | >60% Direction Hit | ✅ Direction accuracy metric |
| MLOps Maturity | >10 Tracked Runs | ✅ MLflow tracking |
| Drift Monitoring | Reports Generated | ✅ Evidently integration |
| Deployment | Docker + API | ✅ FastAPI + Docker |
| Visualization | Dashboard | ✅ Streamlit |

## 🔮 Advanced Features

### GAN for Synthetic Data Augmentation (Optional)

To add GAN-based data augmentation:

1. Create `gan_augmentation.py`
2. Implement TimeGAN or similar architecture
3. Generate synthetic time series data
4. Augment training dataset

Example structure:
```python
class TimeGAN:
    def __init__(self, seq_length, latent_dim):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
    
    def generate_synthetic_data(self, n_samples):
        # Generate synthetic sequences
        pass
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: No data fetched for ticker
```bash
# Solution: Check ticker symbol validity
python -c "import yfinance as yf; print(yf.download('MCC.V', period='1d'))"
```

**Issue**: Model not found
```bash
# Solution: Train model first
python train.py
```

**Issue**: Port already in use
```bash
# Solution: Change port in config.py or kill process
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux
lsof -ti:8000 | xargs kill -9
```

## 📚 References

- [DATACAMP Stock Market Predictions with LSTM in Python](https://www.datacamp.com/tutorial/lstm-python-stock-market)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📝 License

This project is for educational purposes.

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This is a research/educational project. Stock predictions should not be used for actual trading decisions without proper risk assessment and financial advice.
