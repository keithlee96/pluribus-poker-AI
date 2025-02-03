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

## Clustering Parameters

We use optimized clustering parameters based on the [original Pluribus paper](https://www.science.org/doi/10.1126/science.aay2400):

```bash
poker_ai cluster \
  --low_card_rank 10 \
  --high_card_rank 14 \
  --n_river_clusters 100 \
  --n_turn_clusters 100 \
  --n_flop_clusters 100 \
  --n_simulations_river 100 \
  --n_simulations_turn 100 \
  --n_simulations_flop 100 \
  --save_dir research/blueprint_algo
```

These parameters create more sophisticated lookup tables by:
- Using 100 clusters per street (flop/turn/river) for better situation distinction
- Running 100 simulations per situation for more accurate strength estimates
- Using a 20-card deck (T,J,Q,K,A in 4 suits) as in the original Pluribus

## Cloud Deployment

### Requirements

Based on our testing:
- CPU: At least 4 cores (8+ recommended)
- RAM: 8GB minimum (16GB+ recommended)
- Storage: 10GB for code and lookup tables
- Estimated clustering time: 4-8 hours depending on CPU

### Setup on Cloud VM

1. Install Python 3.7+ and git
```bash
sudo apt update
sudo apt install python3-pip python3-venv git
```

2. Clone and setup
```bash
git clone https://github.com/YOUR_USERNAME/pluribus-poker-AI.git
cd pluribus-poker-AI
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
  --n_river_clusters 100 \
  --n_turn_clusters 100 \
  --n_flop_clusters 100 \
  --n_simulations_river 100 \
  --n_simulations_turn 100 \
  --n_simulations_flop 100 \
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
- 100 simulations per situation
- Grouped into 100 clusters

2. Turn stage:
- 190 unique situations
- 100 simulations per situation
- Grouped into 100 clusters

3. River stage:
- 190 unique situations
- 100 simulations per situation
- Grouped into 100 clusters

Total simulated hands during clustering: ~57,000 (190 situations × 100 simulations × 3 stages)

This creates more sophisticated lookup tables than the basic version (which used only 1 cluster per street), allowing the AI to better distinguish between different poker situations during training and gameplay.
