@echo off
REM Quick start script for Stock Prediction System

echo ================================================
echo Stock Price Prediction System - Quick Start
echo ================================================
echo.

:menu
echo Select an option:
echo 1. Setup Check (verify installation)
echo 2. Run Full Pipeline (train models for multiple tickers)
echo 3. Train Single Ticker
echo 4. Start API Server
echo 5. Launch Streamlit Dashboard
echo 6. View MLflow UI
echo 7. Generate Drift Reports
echo 8. Docker Compose Up
echo 9. Docker Compose Down
echo 0. Exit
echo.

set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto setup_check
if "%choice%"=="2" goto run_pipeline
if "%choice%"=="3" goto train_single
if "%choice%"=="4" goto start_api
if "%choice%"=="5" goto start_dashboard
if "%choice%"=="6" goto mlflow_ui
if "%choice%"=="7" goto drift_reports
if "%choice%"=="8" goto docker_up
if "%choice%"=="9" goto docker_down
if "%choice%"=="0" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:setup_check
echo.
echo Running setup check...
python setup_check.py
echo.
pause
goto menu

:run_pipeline
echo.
echo Running full pipeline (3 tickers by default)...
echo This may take 10-20 minutes...
python run_pipeline.py
echo.
pause
goto menu

:train_single
echo.
set /p ticker="Enter ticker symbol (e.g., MCC.V): "
echo Training model for %ticker%...
python train.py
echo.
pause
goto menu

:start_api
echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop
python app.py
pause
goto menu

:start_dashboard
echo.
echo Starting Streamlit dashboard on http://localhost:8501
echo Press Ctrl+C to stop
streamlit run streamlit_app.py
pause
goto menu

:mlflow_ui
echo.
echo Starting MLflow UI on http://localhost:5000
echo Press Ctrl+C to stop
mlflow ui --port 5000
pause
goto menu

:drift_reports
echo.
echo Generating drift reports...
python drift_monitor.py
echo.
echo Reports saved to evidently_reports/
pause
goto menu

:docker_up
echo.
echo Starting Docker containers...
docker-compose up -d
echo.
echo Services running:
echo   - API: http://localhost:8000
echo   - Dashboard: http://localhost:8501
echo   - MLflow: http://localhost:5000
echo.
pause
goto menu

:docker_down
echo.
echo Stopping Docker containers...
docker-compose down
echo.
pause
goto menu

:end
echo.
echo Goodbye!
exit
