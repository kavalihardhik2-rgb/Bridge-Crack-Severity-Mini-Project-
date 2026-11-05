#!/bin/bash
# Setup script for bridgeSSL environment
# Works on HPC clusters with module system

echo "=== Setting up bridgeSSL environment ==="

# Option 1: Use existing PyTorch module (recommended for HPC)
if command -v module &> /dev/null; then
    echo "Detected module system. Loading PyTorch..."
    module load python/3.10-anaconda 2>/dev/null || true
    module load pytorch 2>/dev/null || true
fi

# Check if mamba is available (faster than conda)
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba (faster solver)"
else
    CONDA_CMD="conda"
    echo "Using conda"
fi

# Option 2: Create conda environment
echo ""
echo "Choose setup method:"
echo "  1) pip install (fastest, requires existing PyTorch)"
echo "  2) conda/mamba env create (slower, installs everything)"
echo ""
read -p "Enter choice [1/2]: " choice

case $choice in
    1)
        echo "Installing with pip..."
        pip install -r requirements.txt
        echo "Done! Make sure PyTorch is loaded/installed."
        ;;
    2)
        echo "Creating conda environment..."
        $CONDA_CMD env create -f environment.yml
        echo "Done! Activate with: conda activate bridgessl"
        ;;
    *)
        echo "Invalid choice. Running pip install..."
        pip install -r requirements.txt
        ;;
esac

echo ""
echo "=== Setup complete ==="
echo "Test with: python -c 'import torch; print(torch.cuda.is_available())'"
