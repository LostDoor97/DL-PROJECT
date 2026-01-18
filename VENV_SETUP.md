# Virtual Environment Setup Instructions

## Using Your Existing Virtual Environment

If you have an old project venv in the folder, you can reuse it:

### Windows (PowerShell)

```powershell
# Activate existing venv
.\old_venv\Scripts\Activate.ps1

# Or if you renamed it to venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install/update all requirements
pip install -r requirements.txt

# Verify installation
python setup_check.py
```

### Windows (Command Prompt)

```cmd
# Activate existing venv
old_venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install/update requirements
pip install -r requirements.txt

# Verify
python setup_check.py
```

## Creating a Fresh Virtual Environment

### Windows

```powershell
# Create new venv
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Verify
python setup_check.py
```

### Linux/Mac

```bash
# Create new venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify
python setup_check.py
```

## Conda Environment (Alternative)

```bash
# Create conda environment
conda create -n stock-prediction python=3.10 -y

# Activate
conda activate stock-prediction

# Install requirements
pip install -r requirements.txt

# Verify
python setup_check.py
```

## Troubleshooting

### Issue: TensorFlow Installation Fails

**Windows (CPU-only)**
```bash
pip install tensorflow-cpu==2.15.0
```

**Linux/Mac**
```bash
pip install tensorflow==2.15.0
```

### Issue: Conflicting Dependencies

```bash
# Clear pip cache
pip cache purge

# Install with no-cache
pip install --no-cache-dir -r requirements.txt
```

### Issue: Permission Errors

**Windows**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

## Minimal Requirements for Quick Testing

If full installation is taking too long, install minimal requirements first:

```bash
# Core dependencies only
pip install tensorflow keras numpy pandas yfinance scikit-learn

# Test basic functionality
python -c "from data_pipeline import StockDataPipeline; print('✓ Basic setup OK')"

# Install remaining packages as needed
pip install mlflow fastapi uvicorn streamlit evidently statsmodels pmdarima
```

## GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]==2.15.0

# Verify GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Package Versions

Key package versions in this project:
- Python: 3.10+
- TensorFlow: 2.15.0
- Keras: 2.15.0
- MLflow: 2.9.2
- FastAPI: 0.104.1
- Streamlit: 1.29.0
- Evidently: 0.4.11

## After Setup

Once your environment is ready:

1. **Verify Installation**
   ```bash
   python setup_check.py
   ```

2. **Test Quick Import**
   ```bash
   python -c "import tensorflow; import mlflow; print('✓ All core packages loaded')"
   ```

3. **Run Quick Start Menu**
   ```bash
   # Windows
   start.bat
   
   # Linux/Mac
   chmod +x start.sh
   ./start.sh
   ```

4. **Or Run Pipeline Directly**
   ```bash
   python run_pipeline.py
   ```
