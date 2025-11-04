#!/bin/bash

# Exit on any error
set -e

# Check for flags
FRESH=false
EDITABLE=false

for arg in "$@"; do
    if [[ "$arg" == "--fresh" ]]; then
        FRESH=true
    elif [[ "$arg" == "-e" ]]; then
        EDITABLE=true
    fi
done

echo "Setting up development environment..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is not installed or not in PATH"
    echo "Please install Python 3.12 and try again"
    exit 1
fi

# Handle existing venv based on --fresh flag
if [ -d "venv" ]; then
    if [ "$FRESH" = true ]; then
        echo "Removing existing virtual environment (--fresh flag provided)..."
        rm -rf venv
        echo "Creating new virtual environment with Python 3.12..."
        python3.12 -m venv venv
    else
        echo "Virtual environment already exists. Use --fresh to recreate it."
        echo "Activating existing virtual environment..."
    fi
else
    # Create new virtual environment with Python 3.12
    echo "Creating new virtual environment with Python 3.12..."
    python3.12 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Loop through all directories in "packages" and install the package in development mode with dev dependencies
for package in packages/*; do
    echo "Installing $package in development mode with dev dependencies..."
    if EDITABLE=true; then
        pip install -e $package[dev]
    else
        pip install $package[dev]
    fi
done


echo "Environment setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"

echo "Adding aliases..."
source scripts/aliases.sh