#!/bin/bash
# Quick start script for Stock Prediction System (Linux/Mac)

echo "================================================"
echo "Stock Price Prediction System - Quick Start"
echo "================================================"
echo

show_menu() {
    echo "Select an option:"
    echo "1. Setup Check (verify installation)"
    echo "2. Run Full Pipeline (train models for multiple tickers)"
    echo "3. Train Single Ticker"
    echo "4. Start API Server"
    echo "5. Launch Streamlit Dashboard"
    echo "6. View MLflow UI"
    echo "7. Generate Drift Reports"
    echo "8. Docker Compose Up"
    echo "9. Docker Compose Down"
    echo "0. Exit"
    echo
}

while true; do
    show_menu
    read -p "Enter your choice (0-9): " choice
    
    case $choice in
        1)
            echo
            echo "Running setup check..."
            python setup_check.py
            echo
            read -p "Press Enter to continue..."
            ;;
        2)
            echo
            echo "Running full pipeline (3 tickers by default)..."
            echo "This may take 10-20 minutes..."
            python run_pipeline.py
            echo
            read -p "Press Enter to continue..."
            ;;
        3)
            echo
            read -p "Enter ticker symbol (e.g., MCC.V): " ticker
            echo "Training model for $ticker..."
            python train.py
            echo
            read -p "Press Enter to continue..."
            ;;
        4)
            echo
            echo "Starting FastAPI server on http://localhost:8000"
            echo "Press Ctrl+C to stop"
            python app.py
            ;;
        5)
            echo
            echo "Starting Streamlit dashboard on http://localhost:8501"
            echo "Press Ctrl+C to stop"
            streamlit run streamlit_app.py
            ;;
        6)
            echo
            echo "Starting MLflow UI on http://localhost:5000"
            echo "Press Ctrl+C to stop"
            mlflow ui --port 5000
            ;;
        7)
            echo
            echo "Generating drift reports..."
            python drift_monitor.py
            echo
            echo "Reports saved to evidently_reports/"
            read -p "Press Enter to continue..."
            ;;
        8)
            echo
            echo "Starting Docker containers..."
            docker-compose up -d
            echo
            echo "Services running:"
            echo "  - API: http://localhost:8000"
            echo "  - Dashboard: http://localhost:8501"
            echo "  - MLflow: http://localhost:5000"
            echo
            read -p "Press Enter to continue..."
            ;;
        9)
            echo
            echo "Stopping Docker containers..."
            docker-compose down
            echo
            read -p "Press Enter to continue..."
            ;;
        0)
            echo
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            echo
            ;;
    esac
done
