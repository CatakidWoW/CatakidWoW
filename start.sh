#!/bin/bash

# Financial AI System Startup Script
# This script sets up and starts the Financial AI system

echo "🚀 Starting Financial AI System..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/pyvenv.cfg" ]; then
    echo "❌ Virtual environment activation failed"
    exit 1
fi

echo "✅ Virtual environment activated"

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models logs

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📝 Creating .env file from template..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your API keys before running the system"
        echo "   You can do this later and restart the system"
    else
        echo "⚠️  No .env file found. Please create one with your configuration."
    fi
else
    echo "✅ Configuration file found"
fi

# Start the application
echo "🌐 Starting Financial AI web application..."
echo "=========================================="
echo "📱 The system will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Run the Flask application
python app.py

echo ""
echo "👋 Financial AI system stopped"