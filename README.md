# Autonomous Racing Game AI - Unified Agent

**⚠️ MAJOR UPDATE: Now using unified agent architecture!**

This project uses deep reinforcement learning to train an AI that can autonomously race in Forza Motorsport 7. The agent learns coordinated control of both steering and speed, progressing from complete novice to professional-level racing.

## 🚗 What's New - Unified Architecture

### Previous System (Deprecated)
- ❌ Two independent agents (steering + speed)
- ❌ Agents couldn't coordinate actions
- ❌ Fundamentally broken architecture
- ❌ 28+ critical logic errors

### New Unified System
- ✅ Single agent controlling both steering and speed
- ✅ Coordinated action space (25 combinations)
- ✅ Progressive learning from novice to pro
- ✅ All bugs fixed, proper DQN implementation
- ✅ Curriculum learning with automatic difficulty adjustment
- ✅ Robust crash detection and episode termination
- ✅ Model checkpointing and training recovery

## 📁 Project Structure

### New Unified System Files
```
main_unified.py                    # NEW: Main entry point
UNIFIED_AGENT_GUIDE.md            # Complete usage guide
LOGIC_ERRORS_REPORT.md            # Detailed bug analysis

src/game/
├── unified_agent.py              # Main agent class
├── unified_model.py              # Neural network architectures
├── unified_rewards.py            # Progressive reward system
├── unified_training.py           # Training loop
├── detection.py                  # Fixed chevron detection
└── game_env.py                   # Updated with crash detection
```

### Old Files (Deprecated - Do Not Use)
```
main.py                           # Old dual-agent system
src/game/
├── agent_steering.py             # Deprecated
├── agent_speed.py                # Deprecated
├── agents.py                     # Deprecated
├── training.py                   # Deprecated
└── learning.py                   # Deprecated
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow numpy opencv-python mss vgamepad pygetwindow
```

### Game Setup
1. Launch **Forza Motorsport 7**
2. Set to **Windowed Mode** (not fullscreen)
3. Ensure window title contains "Forza Motorsport 7"

### Run Training
```bash
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

## 🧠 How It Works

### Unified Agent Architecture

The agent uses a **single neural network** that outputs Q-values for all possible (steering, speed) combinations:

```
Action Space:
- Steering: [-1.0, -0.5, 0.0, 0.5, 1.0]
- Speed:    [-1.0, -0.5, 0.0, 0.5, 1.0]
- Total: 5 × 5 = 25 coordinated actions

Example Actions:
  (steering=-1.0, speed=-1.0) → Hard left + full brake
  (steering=0.0, speed=1.0)   → Straight + full throttle
  (steering=0.5, speed=0.5)   → Slight right + moderate acceleration
```

### Deep Q-Learning (DQN)

The agent implements proper DQN with:
- **Experience Replay**: 50,000 experience buffer
- **Target Network**: Updated every 1000 steps for stability
- **Epsilon-Greedy**: Starts at 100% exploration, decays to 1%
- **Bellman Equation**: Correct Q-value updates

### Progressive Learning

The agent learns in stages:

| Stage | Episodes | Focus | Expected Behavior |
|-------|----------|-------|-------------------|
| **Novice** | 0-200 | Stay on track | Jerky but stays on track |
| **Intermediate** | 200-500 | Speed control | Smoother, faster driving |
| **Advanced** | 500-1000 | Optimize lines | Competent racing |
| **Pro** | 1000+ | Master laps | Professional-level |

### Reward System

Multi-component reward that adapts as agent improves:

| Component | Novice Weight | Pro Weight | Purpose |
|-----------|---------------|------------|---------|
| Position | 2.0 | 1.0 | Stay centered on track |
| Speed | 0.5 | 2.0 | Go fast (when safe) |
| Smoothness | 0.3 | 0.7 | Smooth control |
| Progress | 0.2 | 1.5 | Forward movement |
| Crash | -50 | -200 | Avoid crashes |

## 📊 Training Progress Example

```
Episode 1/1000
  Reward: 5.23 (Avg: 5.23, Best: 5.23)
  Steps: 87 | Epsilon: 0.995

Episode 100/1000
  Reward: 78.91 (Avg: 64.32, Best: 82.45)
  Steps: 521 | Epsilon: 0.606

╔════════════════════════════════════╗
║ CURRICULUM ADVANCED TO STAGE 2     ║
╚════════════════════════════════════╝

Episode 500/1000
  Reward: 187.34 (Avg: 145.32, Best: 198.76)
  Steps: 1523 | Epsilon: 0.082
  *** NEW BEST MODEL ***
```

## 🔧 Configuration

### Hyperparameters (in `main_unified.py`)

```python
UnifiedRacingAgent(
    input_dims=6,              # Input features
    mem_size=50000,            # Replay buffer size
    batch_size=64,             # Minibatch size
    eps=1.0,                   # Initial exploration
    eps_min=0.01,              # Minimum exploration
    eps_dec=0.995,             # Epsilon decay per episode
    gamma=0.99,                # Discount factor
    learning_rate=0.001,       # Network learning rate
    target_update_freq=1000    # Target network update frequency
)
```

### Reward Weights (in `unified_rewards.py`)

Adjust to change learning behavior:
```python
reward_weights = {
    'position': 2.0,    # Track positioning
    'speed': 1.0,       # Speed optimization
    'smoothness': 0.5,  # Control smoothness
    'progress': 0.5,    # Forward progress
    'crash': -100.0     # Crash penalty
}
```

## 📚 Documentation

- **[UNIFIED_AGENT_GUIDE.md](UNIFIED_AGENT_GUIDE.md)** - Complete usage guide, troubleshooting, advanced features
- **[LOGIC_ERRORS_REPORT.md](LOGIC_ERRORS_REPORT.md)** - Detailed analysis of 28 bugs that were fixed

## 🐛 Bug Fixes

All 28 logic errors from the original code have been fixed:
1. ✅ Missing learning object initialization
2. ✅ Broken DQN Q-value updates
3. ✅ Hard-coded batch sizes causing crashes
4. ✅ Episodes never terminating (infinite loops)
5. ✅ Array dimension swap in boundary checks
6. ✅ Color sampling from wrong locations
7. ✅ Empty array checks using wrong method
8. ✅ And 21 more...

See [LOGIC_ERRORS_REPORT.md](LOGIC_ERRORS_REPORT.md) for complete details.

## 🎯 Future Enhancements

Potential improvements:
- [ ] Convolutional neural network for raw image input
- [ ] Prioritized experience replay
- [ ] Multi-step returns (n-step DQN)
- [ ] Distributional RL (Rainbow DQN)
- [ ] Continuous action space (DDPG/TD3/SAC)
- [ ] Multi-track training for generalization
- [ ] Opponent avoidance (multi-car racing)
- [ ] Lap time optimization with trajectory planning

## 🤝 Contributing

The codebase is now clean, well-documented, and properly architected. Contributions welcome!

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- Deep Q-Learning: Mnih et al. (2015)
- Dueling DQN: Wang et al. (2016)
- Curriculum Learning: Bengio et al. (2009)

---

**Ready to train a pro-level racing AI? Run `python main_unified.py` and get started! 🏁**
