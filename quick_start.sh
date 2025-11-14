#!/bin/bash
# Quick start script for volatility forecasting project

echo "=========================================="
echo "Volatility Forecasting Project - Quick Start"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "\n[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo -e "\n[1/4] Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n[2/4] Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo -e "\n[3/4] Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run pipeline
echo -e "\n[4/4] Running complete pipeline..."
echo "This may take 50-80 minutes depending on your hardware."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python run_pipeline.py --stage all
    
    # Generate visualizations
    echo -e "\nGenerating visualizations..."
    python visualize_results.py
    
    echo -e "\n=========================================="
    echo "✓ Pipeline complete!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - results/preds/       (forecasts)"
    echo "  - results/tables/      (metrics)"
    echo "  - results/figs/        (plots)"
    echo "  - results/backtests/   (portfolio results)"
else
    echo "Pipeline cancelled."
fi

