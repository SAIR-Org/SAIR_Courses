#!/bin/bash

echo "🚀 Setting up DNN Training Pipeline..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data models results logs

echo "✅ Setup complete!"
echo ""
echo "Run the pipeline:"
echo "  python run_pipeline.py --mode all"