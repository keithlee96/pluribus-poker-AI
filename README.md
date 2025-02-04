# Pluribus Poker AI - Optimized Version

This is a fork of the [original Pluribus Poker AI implementation](https://github.com/fedden/poker_ai) with optimized clustering parameters and cloud deployment instructions.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/HubertPiotrowski/pluribus-poker-AI.git
cd pluribus-poker-AI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## GPU Acceleration Support

The clustering process now supports GPU acceleration for significantly faster computation. This is particularly beneficial when working with large numbers of simulations and clusters.

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or 12.x installed
- Python 3.7+

### Setup GPU Support

1. Make sure you have CUDA toolkit installed. You can download it from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

2. Run the setup script:
   ```bash
   chmod +x setup_gpu.sh
   ./setup_gpu.sh
   ```

The script will:
- Check for CUDA availability
- Install required GPU libraries (cupy and cuml)
- Set up a virtual environment with all dependencies
- Test the GPU setup

The system will automatically use GPU acceleration when available, falling back to CPU if:
- GPU is not available
- GPU memory is insufficient
- Any GPU-related error occurs

## Clustering Parameters

We use optimized clustering parameters based on the [original Pluribus paper](https://www.science.org/doi/10.1126/science.aay2400):

```bash
poker_ai cluster \
  --low_card_rank 10 \
  --high_card_rank 14 \
  --n_river_clusters 1000 \
  --n_turn_clusters 1000 \
  --n_flop_clusters 1000 \
  --n_simulations_river 1000 \
  --n_simulations_turn 1000 \
  --n_simulations_flop 1000 \
  --save_dir research/blueprint_algo
```

These parameters create more sophisticated lookup tables by:
- Using 1000 clusters per street (flop/turn/river) for better situation distinction
- Running 1000 simulations per situation for more accurate strength estimates
- Using a 20-card deck (T,J,Q,K,A in 4 suits) as in the original Pluribus

## Cloud Deployment

### Requirements

Based on our testing:
- GPU: NVIDIA GPU with 8GB+ VRAM (recommended for large clusters)
- CPU: At least 4 cores (8+ recommended if not using GPU)
- RAM: 16GB minimum (32GB+ recommended)
- Storage: 10GB for code and lookup tables
- Estimated clustering time: 
  * With GPU: 1-2 hours
  * CPU only: 4-8 hours

### Setup on Cloud VM

1. Install Python 3.7+, git, and CUDA (if using GPU)
```bash
sudo apt update
sudo apt install python3-pip python3-venv git

# For GPU support, install CUDA toolkit
# Visit https://developer.nvidia.com/cuda-downloads for latest instructions
```

2. Clone and setup
```bash
git clone https://github.com/YOUR_USERNAME/pluribus-poker-AI.git
cd pluribus-poker-AI

# For GPU support
chmod +x setup_gpu.sh
./setup_gpu.sh

# For CPU only
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

3. Run clustering (consider using screen/tmux as it takes several hours)
```bash
screen -S poker  # Create new screen session
mkdir -p research/blueprint_algo
poker_ai cluster \
  --low_card_rank 10 \
  --high_card_rank 14 \
  --n_river_clusters 1000 \
  --n_turn_clusters 1000 \
  --n_flop_clusters 1000 \
  --n_simulations_river 1000 \
  --n_simulations_turn 1000 \
  --n_simulations_flop 1000 \
  --save_dir research/blueprint_algo
# Ctrl+A, D to detach from screen
```

4. Check progress
```bash
screen -r poker  # Reattach to screen session
```

### Training (After Clustering)

Once clustering is complete, you can start training:
```bash
poker_ai train start
```

### Playing Against the AI

After training completes:
```bash
poker_ai play
```

## Performance Notes

The clustering process creates lookup tables that group similar poker situations together. With our parameters:

1. Flop stage:
- 190 unique situations
- 1000 simulations per situation
- Grouped into 1000 clusters

2. Turn stage:
- 190 unique situations
- 1000 simulations per situation
- Grouped into 1000 clusters

3. River stage:
- 190 unique situations
- 1000 simulations per situation
- Grouped into 1000 clusters

Total simulated hands during clustering: ~570,000 (190 situations × 1000 simulations × 3 stages)

This creates more sophisticated lookup tables than the basic version, allowing the AI to better distinguish between different poker situations during training and gameplay. The GPU acceleration significantly speeds up the clustering process, making it practical to use larger numbers of clusters and simulations.
