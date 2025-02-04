#!/bin/bash

echo "Setting up GPU support for Pluribus Poker AI..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Please install CUDA toolkit first."
    echo "Visit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

echo "Detected CUDA version: $CUDA_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install base requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install appropriate CUDA version of cupy and cuml
if [ $CUDA_MAJOR -eq 11 ]; then
    echo "Installing CUDA 11.x compatible packages..."
    pip install cupy-cuda11x
    pip install cuml-cuda11
elif [ $CUDA_MAJOR -eq 12 ]; then
    echo "Installing CUDA 12.x compatible packages..."
    pip install cupy-cuda12x
    pip install cuml-cuda12
else
    echo "Unsupported CUDA version. Please use CUDA 11.x or 12.x"
    exit 1
fi

echo "Testing GPU setup..."
python3 - << EOF
try:
    import cupy as cp
    import cuml
    print("GPU support successfully installed!")
    print("Available GPU memory:", cp.get_default_memory_pool().used_bytes() / 1024**3, "GB")
except ImportError as e:
    print("Error importing GPU libraries:", e)
EOF

echo "Setup complete! You can now use GPU acceleration for clustering."
echo "Run your clustering command as usual - GPU will be used automatically if available."