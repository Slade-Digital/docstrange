#!/bin/bash

# Setup script for docstrange
# This script creates a virtual environment and installs all dependencies

echo "🚀 Setting up docstrange environment..."

# Check if Python 3 is available
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install all dependencies
echo "📚 Installing dependencies..."
pip install -e .

# Install additional dependencies for full functionality
echo "🔧 Installing additional dependencies..."
pip install requests beautifulsoup4 pandas openpyxl python-pptx Pillow lxml

# Install PaddleOCR (optional but recommended for OCR)
echo "🤖 Installing PaddleOCR for OCR capabilities..."
pip install paddlepaddle paddleocr

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To test the library, run:"
echo "python test_enhanced_library.py"
echo ""
echo "To install the library in development mode, run:"
echo "pip install -e ." 