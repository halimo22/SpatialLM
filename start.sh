#!/bin/bash
set -e

echo "🚀 Starting SpatialLM FastAPI server..."
echo "📁 Working directory: $(pwd)"

# Source conda
source /opt/conda/etc/profile.d/conda.sh
conda activate spatiallm

echo "🐍 Python version: $(python --version)"

# Check FastAPI installation
echo "📦 Checking FastAPI installation..."
python -c "import fastapi; print('FastAPI version:', fastapi.__version__)" || echo "❌ FastAPI not found"

# Check required files
echo "🔍 Checking required files..."
ls -la app.py || echo "❌ app.py not found"
ls -la models/ || echo "❌ models directory not found"
ls -la code_template.txt || echo "❌ code_template.txt not found"

echo "🏃 Starting uvicorn..."
exec uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info