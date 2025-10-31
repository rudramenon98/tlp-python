#!/bin/bash

# Exit on any error
set -e

# Check for --force flag
FORCE=false
if [[ "$1" == "--force" ]]; then
    FORCE=true
fi

echo "Setting up development environment..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is not installed or not in PATH"
    echo "Please install Python 3.12 and try again"
    exit 1
fi

# Handle existing venv based on --force flag
if [ -d "venv" ]; then
    if [ "$FORCE" = true ]; then
        echo "Removing existing virtual environment (--force flag provided)..."
        rm -rf venv
        echo "Creating new virtual environment with Python 3.12..."
        python3.12 -m venv venv
    else
        echo "Virtual environment already exists. Use --force to recreate it."
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

# Install the package in development mode with dev dependencies
echo "Installing package in development mode with dev dependencies..."
pip install -e .[dev]

# Install local requirements if the file exists
if [ -f "requirements.local.txt" ]; then
    echo "Installing local requirements..."
    pip install -r requirements.local.txt
else
    echo "Warning: requirements.local.txt not found, skipping local requirements"
fi

echo "Environment setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"



# add aliases for format-py and setup-env
alias format-py="scripts/format_py.sh"
alias setup-env="scripts/setup_environment.sh"