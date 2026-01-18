# ✅ PROJECT CHECKLIST

Use this checklist to get your LSTM Stock Prediction system up and running!

## 📋 Pre-Flight Checklist

### Environment Setup
- [ ] Python 3.10+ installed
  - Check: `python --version`
- [ ] Virtual environment ready (or old venv available)
  - Create: `python -m venv venv`
  - Activate (Windows): `.\venv\Scripts\Activate.ps1`
  - Activate (Linux/Mac): `source venv/bin/activate`
- [ ] Dependencies installed
  - Run: `pip install -r requirements.txt`
- [ ] Setup verified
  - Run: `python setup_check.py`

### Directory Structure
- [ ] All directories created automatically by setup_check.py
  - `data/raw/` ✓
  - `data/processed/` ✓
  - `models/` ✓
  - `logs/` ✓
  - `mlruns/` ✓
  - `evidently_reports/` ✓

## 🚀 First Run Checklist

### Option A: Quick Start (Recommended)
- [ ] Run quick start menu
  - Windows: `start.bat`
  - Linux/Mac: `chmod +x start.sh && ./start.sh`
- [ ] Select option 1 (Setup Check)
- [ ] Select option 2 (Run Full Pipeline)
- [ ] Wait 10-20 minutes for training
- [ ] Select option 6 (View MLflow UI)
- [ ] Select option 5 (Launch Dashboard)

### Option B: Manual Step-by-Step
- [ ] **Step 1: Train First Model**
  ```bash
  python train.py
  ```
  - Expected: Creates model in `models/` folder
  - Expected: Creates MLflow run in `mlruns/`
  - Expected: Shows direction accuracy >60%

- [ ] **Step 2: Verify Model**
  ```python
  from predict import find_latest_model
  print(find_latest_model('MCC.V'))
  ```
  - Expected: Shows path to .h5 file

- [ ] **Step 3: Generate Predictions**
  ```bash
  python predict.py
  ```
  - Expected: Shows prediction values
  - Expected: No errors

- [ ] **Step 4: Train Multiple Tickers**
  ```bash
  python run_pipeline.py
  ```
  - Expected: Trains 3 models (default)
  - Expected: Shows comparison results

- [ ] **Step 5: Compare with Baseline**
  ```bash
  python baseline_arima.py
  ```
  - Expected: LSTM vs ARIMA comparison
  - Expected: Both models logged to MLflow

- [ ] **Step 6: Start API Server**
  ```bash
  python app.py
  ```
  - Expected: Server runs on http://localhost:8000
  - Expected: Can access http://localhost:8000/docs

- [ ] **Step 7: Test API**
  ```bash
  curl "http://localhost:8000/predict?ticker=MCC.V&steps=30"
  ```
  - Expected: JSON response with predictions

- [ ] **Step 8: Launch Dashboard**
  ```bash
  streamlit run streamlit_app.py
  ```
  - Expected: Opens browser to http://localhost:8501
  - Expected: Can see predictions

- [ ] **Step 9: Generate Drift Reports**
  ```bash
  python drift_monitor.py
  ```
  - Expected: Creates HTML files in `evidently_reports/`

- [ ] **Step 10: View MLflow**
  ```bash
  mlflow ui
  ```
  - Expected: Opens http://localhost:5000
  - Expected: Can see all runs

## 🐳 Docker Deployment Checklist

### Prerequisites
- [ ] Docker installed
  - Check: `docker --version`
- [ ] Docker Compose installed
  - Check: `docker-compose --version`

### Deployment Steps
- [ ] **Build Images**
  ```bash
  docker-compose build
  ```
  - Expected: Builds successfully without errors

- [ ] **Start Services**
  ```bash
  docker-compose up -d
  ```
  - Expected: 3 containers running (api, dashboard, mlflow)

- [ ] **Verify Services**
  - [ ] API: http://localhost:8000
  - [ ] Dashboard: http://localhost:8501
  - [ ] MLflow: http://localhost:5000

- [ ] **Check Logs**
  ```bash
  docker-compose logs -f
  ```
  - Expected: No error messages

- [ ] **Test API in Docker**
  ```bash
  curl "http://localhost:8000/health"
  ```
  - Expected: `{"status":"healthy"}`

## 📊 Validation Checklist

### Model Performance
- [ ] Direction accuracy >60% achieved
  - Check in MLflow UI or training logs
- [ ] At least 10 MLflow runs logged
  - Check: Count runs in MLflow UI
- [ ] Models saved as .h5 files
  - Check: `dir models\*.h5` or `ls models/*.h5`

### MLOps Pipeline
- [ ] MLflow tracking working
  - Evidence: Can view runs in UI
- [ ] Docker containers running
  - Check: `docker ps`
- [ ] API endpoints responding
  - Test: http://localhost:8000/docs
- [ ] Dashboard displays predictions
  - Test: Open Streamlit and generate predictions
- [ ] Drift reports generated
  - Check: Files in `evidently_reports/`

### CI/CD (Optional)
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] GitHub Actions workflow runs
  - Check: Actions tab in GitHub
- [ ] CI/CD pipeline passes
  - Expected: Green checkmark

## 🎯 Deliverables Checklist

Required for submission:

- [ ] **GitHub Repository**
  - [ ] All code committed
  - [ ] README.md included
  - [ ] .gitignore properly configured
  - [ ] Requirements.txt included

- [ ] **MLflow Runs**
  - [ ] >10 runs tracked
  - [ ] Best model identified
  - [ ] Metrics logged (MSE, direction accuracy)
  - [ ] Model artifacts saved

- [ ] **Docker Image**
  - [ ] Dockerfile present
  - [ ] docker-compose.yml configured
  - [ ] Successfully builds
  - [ ] Successfully runs

- [ ] **Streamlit Dashboard**
  - [ ] Shows actual vs predicted curves
  - [ ] Interactive controls work
  - [ ] Can download predictions
  - [ ] Displays MLflow runs

- [ ] **Drift Reports**
  - [ ] HTML reports generated
  - [ ] Data drift visualized
  - [ ] Model performance tracked

## 🐛 Troubleshooting Checklist

If something doesn't work:

- [ ] **Python version correct?**
  - Run: `python --version`
  - Need: 3.10 or higher

- [ ] **All dependencies installed?**
  - Run: `pip list`
  - Check for: tensorflow, mlflow, fastapi, streamlit

- [ ] **Directories exist?**
  - Run: `python setup_check.py`

- [ ] **Port conflicts?**
  - Change ports in `config.py`
  - Or kill processes using ports

- [ ] **Model not found?**
  - Train a model first: `python train.py`

- [ ] **Data fetch fails?**
  - Check internet connection
  - Verify ticker symbol is valid

## 📈 Success Indicators

You'll know it's working when:

✅ Training completes without errors  
✅ Direction accuracy >60% shown  
✅ MLflow UI shows your runs  
✅ API returns predictions  
✅ Dashboard displays charts  
✅ Docker containers all running  
✅ Drift reports are generated  

## 🎓 Next Level Checklist (Advanced)

Once basic system works:

- [ ] **Experiment with Hyperparameters**
  - [ ] Try different LSTM units
  - [ ] Adjust dropout rates
  - [ ] Change batch sizes

- [ ] **Add More Tickers**
  - [ ] Edit `config.py`
  - [ ] Train on new tickers
  - [ ] Compare performance

- [ ] **Implement GAN Augmentation** (Optional)
  - [ ] Create GAN architecture
  - [ ] Generate synthetic data
  - [ ] Train with augmented data

- [ ] **Optimize Performance**
  - [ ] Profile code for bottlenecks
  - [ ] Implement caching
  - [ ] Use GPU if available

- [ ] **Production Deployment**
  - [ ] Deploy to cloud (AWS/GCP/Azure)
  - [ ] Set up monitoring
  - [ ] Configure auto-scaling

## ✨ Final Check

- [ ] All core features working
- [ ] Documentation read and understood
- [ ] Can train models on demand
- [ ] Can generate predictions
- [ ] Can view results in dashboard
- [ ] Can explain the architecture
- [ ] Ready to present/submit

---

**Congratulations! 🎉**

If you've checked all the boxes, you have a fully functional LSTM stock prediction system with complete MLOps pipeline!

**Need Help?**
- See `README.md` for detailed instructions
- See `QUICK_START.md` for command reference  
- See `VENV_SETUP.md` for environment issues
- See `PROJECT_SUMMARY.md` for architecture details
