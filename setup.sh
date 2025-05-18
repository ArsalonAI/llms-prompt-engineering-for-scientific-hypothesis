#!/bin/bash

echo "Setting up LLMs for Scientific Hypothesis Generation project..."

# Create virtual environment if it doesn't exist
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

echo "Creating fresh virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Clean up any existing installations
echo "Cleaning up existing packages..."
pip freeze | xargs pip uninstall -y

# Install dependencies with verbose output
echo "Installing dependencies..."
pip install -r requirements.txt -v

# Verify plotly installation
echo "Verifying plotly installation..."
if python -c "import plotly" &> /dev/null; then
    echo "✓ Plotly successfully installed"
else
    echo "! Error: Plotly installation failed. Trying alternative install..."
    pip install plotly --no-cache-dir -v
fi

# Install IPython kernel for Jupyter
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=llm_hypothesis --display-name="LLM Hypothesis Generation"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating empty .env file..."
    touch .env
    echo "Please add your API keys to the .env file"
fi

echo "Setup complete! Please follow these next steps:"
echo "1. Add required API keys to .env file (OPENAI_API_KEY, TOGETHER_API_KEY, WANDB_API_KEY)"
echo "2. Activate the virtual environment with: source venv/bin/activate"
echo "3. Start jupyter notebook with: jupyter notebook"