# Unified Racing Agent - Complete Guide

## Overview

This document describes the **new unified racing agent architecture** that replaces the old dual-agent system. The unified agent controls both steering and speed simultaneously, enabling it to learn coordinated racing behavior from complete novice to professional level.

---

## Table of Contents

1. [Why Unified Architecture?](#why-unified-architecture)
2. [Architecture Overview](#architecture-overview)
3. [Key Components](#key-components)
4. [Getting Started](#getting-started)
5. [Training Process](#training-process)
6. [Progressive Learning](#progressive-learning)
7. [File Structure](#file-structure)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Why Unified Architecture?

### The Problem with Dual Agents

The old system used two independent agents:
- **Steering Agent**: Controlled only steering
- **Speed Agent**: Controlled only throttle/brake

**This was fundamentally broken** because:

```
Steering Agent        Speed Agent
     ↓                     ↓
  Steers                Accelerates
  (no movement)         (can't turn)
     ↓                     ↓
       CAR CRASHES
```

**Problems:**
1. **No coordination**: Agents couldn't communicate or learn together
2. **Impossible tasks**: Steering without moving or moving without steering
3. **Ambiguous rewards**: Which agent caused a crash?
4. **Can't learn racing**: Racing requires "slow down BEFORE sharp turns" - neither agent could learn this

### The Unified Solution

The new unified agent outputs **both steering and speed as a coordinated action**:

```
Unified Racing Agent
        ↓
   Neural Network
        ↓
  (steering, speed)  ← Single coordinated decision
        ↓
    Controller
        ↓
  Car drives properly
```

**Benefits:**
1. ✅ **Coordinated control**: Learns to steer AND control speed together
2. ✅ **Realistic racing**: Can learn "brake before turn" strategies
3. ✅ **Clear rewards**: Single agent gets single reward for its complete action
4. ✅ **Better learning**: More stable, faster convergence

---

## Architecture Overview

### Action Space

The agent uses a **discrete action grid** combining steering and speed:

```
Steering values: [-1.0, -0.5, 0.0, 0.5, 1.0]  (5 options)
Speed values:    [-1.0, -0.5, 0.0, 0.5, 1.0]  (5 options)

Total actions: 5 × 5 = 25 combinations

Examples:
  Action 0:  (steering=-1.0, speed=-1.0) → Hard left + full brake
  Action 12: (steering=0.0, speed=0.5)   → Straight + moderate throttle
  Action 24: (steering=1.0, speed=1.0)   → Hard right + full throttle
```

### Neural Network

```
Input Layer (6 features)
    ↓
Dense (512 neurons, ReLU)
Batch Normalization
Dropout (20%)
    ↓
Dense (512 neurons, ReLU)
Batch Normalization
Dropout (20%)
    ↓
Dense (256 neurons, ReLU)
Batch Normalization
Dropout (20%)
    ↓
Dense (256 neurons, ReLU)
Batch Normalization
    ↓
Output Layer (25 Q-values)
```

**Advanced Option**: Dueling DQN architecture available in `unified_model.py`

### Deep Q-Learning (DQN)

The agent uses proper DQN with:
- **Experience Replay**: Stores and learns from past experiences
- **Target Network**: Stabilizes learning by using a separate target network
- **Epsilon-Greedy**: Balances exploration (random actions) and exploitation (best known actions)
- **Bellman Equation**: Correctly updates Q-values for learning

```python
# Proper Q-learning update (now fixed!)
Q_target[action] = reward + gamma * max(Q_next_state) * (1 - done)
```

---

## Key Components

### 1. `unified_agent.py`
Main agent class with:
- Action selection (epsilon-greedy)
- Experience storage
- DQN learning algorithm
- Model checkpointing
- State tracking

### 2. `unified_model.py`
Neural network architectures:
- Standard DQN
- Dueling DQN (advanced)
- Configurable layers and hyperparameters

### 3. `unified_rewards.py`
Reward system with multiple components:

| Component | Purpose | Weight (Novice) | Weight (Pro) |
|-----------|---------|-----------------|--------------|
| **Position** | Stay centered on track | 2.0 | 1.0 |
| **Speed** | Go fast on straights, slow in turns | 0.5 | 2.0 |
| **Smoothness** | Discourage jerky control | 0.3 | 0.7 |
| **Progress** | Encourage forward movement | 0.2 | 1.5 |
| **Crash** | Penalty for crashing | -50 | -200 |

**Progressive rewards** automatically adjust as agent improves!

### 4. `unified_training.py`
Complete training loop:
- Episode management
- Progress tracking
- Checkpointing every 50 episodes
- Performance statistics
- Automatic difficulty adjustment

### 5. `main_unified.py`
User interface:
- Train new agent
- Continue from checkpoint
- Evaluate trained agent
- Quick training (100 episodes)
- Full training (1000 episodes)

---

## Getting Started

### Prerequisites

```bash
# Required packages
pip install tensorflow numpy opencv-python mss vgamepad pygetwindow
```

### Game Setup

1. **Launch Forza Motorsport 7**
2. **Set to Windowed Mode** (not fullscreen)
3. **Position window** so it's visible and accessible
4. **Verify game window title** contains "Forza Motorsport 7"

### Quick Start

```bash
# Run the unified agent trainer
python main_unified.py
```

**Menu Options:**
1. Train New Agent (Novice to Pro)
2. Continue Training from Checkpoint
3. Evaluate Trained Agent
4. Quick Training (100 episodes)
5. Full Training (1000 episodes)
6. System Info & Configuration
7. Exit

---

## Training Process

### Training Stages

The agent learns progressively through multiple stages:

#### Stage 1: Novice (Episodes 0-200)
- **Goal**: Learn to stay on track
- **Focus**: Position rewards heavily weighted
- **Expected behavior**: Jerky but stays on track

#### Stage 2: Intermediate (Episodes 200-500)
- **Goal**: Learn speed control
- **Focus**: Balance position and speed
- **Expected behavior**: Smoother control, starts going faster

#### Stage 3: Advanced (Episodes 500-1000)
- **Goal**: Optimize racing lines
- **Focus**: All components balanced
- **Expected behavior**: Competent racing, optimizing turns

#### Stage 4: Pro (Episodes 1000+)
- **Goal**: Master lap times
- **Focus**: Maximum performance
- **Expected behavior**: Professional-level racing

### Curriculum Learning

With curriculum learning enabled, the agent automatically advances through stages based on performance:

```python
Stage 1 → Stage 2: Avg reward > 30 for 50 episodes
Stage 2 → Stage 3: Avg reward > 80 for 100 episodes
Stage 3 → Stage 4: Avg reward > 150 for 150 episodes
```

### Training Example

```bash
$ python main_unified.py

Choose option: 1 (Train New Agent)
Number of episodes: 1000

Episode 1/1000 [15.2s]
  Reward: 5.23 (Avg: 5.23, Best: 5.23)
  Steps: 87 (Avg: 87.0)
  Epsilon: 0.9950
  Memory size: 87/50000

Episode 50/1000 [18.5s]
  Reward: 45.67 (Avg: 32.15, Best: 51.23)
  Steps: 342 (Avg: 215.3)
  Epsilon: 0.7788
  Memory size: 10750/50000
  *** NEW BEST MODEL (Reward: 51.23) ***

Episode 100/1000 [22.1s]
  Reward: 78.91 (Avg: 64.32, Best: 82.45)
  Steps: 521 (Avg: 398.7)
  Epsilon: 0.6065

╔════════════════════════════════════╗
║ CURRICULUM ADVANCED TO STAGE 2     ║
╚════════════════════════════════════╝
```

### Checkpointing

Automatic saves every 50 episodes:
```
models/unified_agent/
  ├── checkpoint_episode_50.h5
  ├── checkpoint_episode_100.h5
  ├── best_model.h5
  ├── replay_buffer_episode_50.pkl
  └── metrics_episode_50.pkl
```

---

## Progressive Learning

### How It Works

The reward function automatically adjusts based on agent performance:

```python
# Novice weights (episodes 0-200)
weights = {
    'position': 2.0,   # Heavy focus on staying on track
    'speed': 0.5,      # Light speed encouragement
    'smoothness': 0.3,
    'progress': 0.2
}

# Pro weights (episodes 1000+)
weights = {
    'position': 1.0,
    'speed': 2.0,      # Heavy focus on speed
    'smoothness': 0.7, # Encourage smooth control
    'progress': 1.5    # Reward progress heavily
}
```

### Monitoring Progress

The training loop prints detailed statistics:

```
TRAINING SUMMARY - Episode 500
────────────────────────────────────────

Recent Performance (Last 100 episodes):
  Average reward: 145.32 ± 23.45
  Max reward: 198.76
  Min reward: 87.23
  Average length: 1234.5 steps

Overall Performance:
  Average reward: 98.54
  Best reward: 198.76 (Episode 487)
  Average length: 876.3 steps

Learning Progress:
  First 100 episodes avg: 45.23
  Last 100 episodes avg: 145.32
  Improvement: +221.4%
────────────────────────────────────────
```

---

## File Structure

### New Files Created

```
src/game/
├── unified_agent.py         # Main agent class
├── unified_model.py         # Neural network models
├── unified_rewards.py       # Reward system
└── unified_training.py      # Training loop

main_unified.py             # New main entry point
UNIFIED_AGENT_GUIDE.md      # This file
LOGIC_ERRORS_REPORT.md      # Bug analysis report
```

### Files Modified

```
src/game/
├── detection.py            # Fixed array indexing bugs
└── game_env.py             # Added crash detection
```

### Files Deprecated

The old dual-agent files are kept for reference but not used:
```
src/game/
├── agent_steering.py       # Old (do not use)
├── agent_speed.py          # Old (do not use)
├── agents.py               # Old (do not use)
├── training.py             # Old (do not use)
└── learning.py             # Old (do not use)

main.py                     # Old (do not use)
```

---

## Configuration

### Hyperparameters

Edit `main_unified.py` to adjust training parameters:

```python
agent = UnifiedRacingAgent(
    game_env=game_env,
    input_dims=6,              # Number of input features
    mem_size=50000,            # Replay buffer size
    batch_size=64,             # Minibatch size
    eps=1.0,                   # Initial exploration rate
    eps_min=0.01,              # Minimum exploration rate
    eps_dec=0.995,             # Epsilon decay per episode
    gamma=0.99,                # Discount factor
    learning_rate=0.001,       # Neural network learning rate
    target_update_freq=1000,   # Target network update frequency
)
```

### Reward Tuning

Edit `unified_rewards.py` to adjust reward weights:

```python
self.reward_weights = {
    'position': 2.0,    # How much to reward staying centered
    'speed': 1.0,       # How much to reward going fast
    'smoothness': 0.5,  # How much to reward smooth control
    'progress': 0.5,    # How much to reward forward progress
    'crash': -100.0     # How much to penalize crashes
}
```

### Training Options

```python
train_racing_agent(
    agent,
    num_episodes=1000,      # Total episodes
    save_frequency=50,      # Save checkpoint every N episodes
    verbose=True,           # Print detailed info
    use_curriculum=True,    # Enable curriculum learning
    visualization=True      # Show game overlays
)
```

---

## Troubleshooting

### Common Issues

#### 1. "Game window not detected"

**Solution:**
- Ensure game is running
- Set game to windowed mode (not fullscreen)
- Check window title contains "Forza Motorsport 7"
- Run `gw.getAllTitles()` in Python to see all window titles

#### 2. "Agent not learning / Reward stays low"

**Possible causes:**
- Chevron detection not working (check templates in `src/data/`)
- Reward function too harsh (adjust weights)
- Learning rate too high/low
- Episode terminating too early

**Debug:**
```python
# Enable verbose mode to see step-by-step info
train_racing_agent(agent, verbose=True)
```

#### 3. "Training very slow"

**Solutions:**
- Enable GPU acceleration (check with `tf.config.list_physical_devices('GPU')`)
- Reduce `mem_size` (smaller replay buffer)
- Reduce `batch_size`
- Disable visualization

#### 4. "IndexError / Array dimension errors"

**These should be fixed!** The new code has all array indexing bugs corrected. If you still see these:
- Ensure you're using `main_unified.py`, not old `main.py`
- Check that `detection.py` has the fixes applied
- Verify `game_env.py` has updated `is_done()` method

#### 5. "Episode never ends"

**Fixed!** New crash detection in `game_env.py`:
- Detects no chevrons for 30 frames → episode ends
- Maximum 5000 steps per episode → prevents infinite loops

### Performance Optimization

For **faster training**:

```python
# Use smaller network
model = Sequential([
    Dense(256, ...),
    Dense(256, ...),
    Dense(n_actions, ...)
])

# Smaller replay buffer
mem_size=10000

# Less frequent target updates
target_update_freq=500
```

For **better learning**:

```python
# Larger replay buffer
mem_size=100000

# Slower epsilon decay
eps_dec=0.999

# More frequent target updates
target_update_freq=2000
```

---

## Advanced Features

### Dueling DQN

For better performance, use Dueling DQN architecture:

```python
# In unified_model.py
from unified_model import build_dueling_dqn_model

# In unified_agent.py, replace:
self.q_eval = build_dueling_dqn_model(input_dims, self.n_actions, learning_rate)
```

### Custom Curriculum

Create your own curriculum stages:

```python
class MyCustomCurriculum(UnifiedRewardSystem):
    def __init__(self, monitor_width):
        super().__init__(monitor_width, difficulty='progressive')

    def advance_curriculum(self, episode, avg_reward):
        if episode == 100:
            print("Moving to Stage 2!")
            self.weights['speed'] = 1.5

        if episode == 300:
            print("Moving to Stage 3!")
            self.weights['position'] = 1.0
            self.weights['speed'] = 2.0
```

### Transfer Learning

Resume training from a checkpoint with modified hyperparameters:

```python
agent = UnifiedRacingAgent(...)
agent.load_model('models/unified_agent/checkpoint_episode_500.h5')

# Fine-tune with lower learning rate
agent.learning_rate = 0.0001

train_racing_agent(agent, num_episodes=500)
```

---

## Summary

### Key Improvements

✅ **Unified architecture** - Single agent controls both steering and speed
✅ **All bugs fixed** - 28 logic errors corrected (see LOGIC_ERRORS_REPORT.md)
✅ **Progressive learning** - Novice to pro curriculum
✅ **Proper DQN** - Correct Q-learning implementation
✅ **Crash detection** - Episodes actually terminate
✅ **Checkpointing** - Save and resume training
✅ **Performance tracking** - Detailed metrics and statistics

### Next Steps

1. **Train the agent**: Run `python main_unified.py`
2. **Monitor progress**: Watch rewards increase over episodes
3. **Evaluate**: Test trained agent with option 3
4. **Iterate**: Adjust rewards/hyperparameters as needed

### Support

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review [LOGIC_ERRORS_REPORT.md](LOGIC_ERRORS_REPORT.md) for technical details
3. Check code comments in source files

---

**Happy Training! 🏎️💨**
