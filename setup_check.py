"""
Quick setup script to verify installation and create directories
"""
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    return True


def check_dependencies():
    """Check if key dependencies are installed"""
    required_packages = [
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'yfinance',
        'mlflow',
        'fastapi',
        'streamlit',
        'evidently'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    return len(missing) == 0, missing


def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'mlruns',
        'evidently_reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")


def test_imports():
    """Test importing main modules"""
    modules = [
        'config',
        'data_pipeline',
        'model',
        'train',
        'predict',
        'app'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py imports successfully")
        except Exception as e:
            print(f"❌ Error importing {module}: {e}")
            return False
    
    return True


def main():
    print("="*70)
    print("Stock Prediction System - Setup Verification")
    print("="*70 + "\n")
    
    # Check Python version
    print("1. Checking Python version...")
    if not check_python_version():
        return
    print()
    
    # Check dependencies
    print("2. Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
        return
    print()
    
    # Create directories
    print("3. Creating directories...")
    create_directories()
    print()
    
    # Test imports
    print("4. Testing module imports...")
    if not test_imports():
        print("\n❌ Some modules failed to import")
        return
    print()
    
    # Success
    print("="*70)
    print("✅ Setup verification completed successfully!")
    print("="*70 + "\n")
    
    print("Quick Start Commands:")
    print("  1. Train models:        python run_pipeline.py")
    print("  2. Single ticker train: python train.py")
    print("  3. Start API server:    python app.py")
    print("  4. Launch dashboard:    streamlit run streamlit_app.py")
    print("  5. View MLflow UI:      mlflow ui")
    print("  6. Docker deployment:   docker-compose up -d")
    print()
    
    print("For detailed instructions, see README.md")


if __name__ == "__main__":
    main()
