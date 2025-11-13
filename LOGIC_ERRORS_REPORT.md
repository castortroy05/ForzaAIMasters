# ForzaAIMasters - Logic Errors & Architecture Assessment Report

**Generated:** 2025-11-13
**Project:** Autonomous Racing Game AI
**Total Issues Found:** 28 (including 1 critical architectural flaw)

---

## EXECUTIVE SUMMARY

This report documents a comprehensive analysis of the ForzaAIMasters codebase. The analysis revealed **28 significant issues** including:

- **1 CRITICAL ARCHITECTURAL FLAW** that makes the system fundamentally broken
- **7 CRITICAL runtime bugs** that prevent training from working
- **5 HIGH PRIORITY issues** affecting learning effectiveness
- **13 MEDIUM PRIORITY bugs** impacting performance and reliability
- **2 MINOR code quality issues**

### Most Critical Finding

**The dual-agent architecture is fundamentally flawed.** The system uses two independent agents (steering and speed) that operate without coordination. This means:
- The steering agent steers but doesn't control movement (car doesn't move)
- The speed agent accelerates but can't steer (car goes straight into walls)
- Neither agent can learn the coupled relationship between steering and speed
- Reward signals are ambiguous (which agent caused a crash?)

**Recommendation:** Redesign to a single unified agent with multi-dimensional action space controlling both steering and speed simultaneously.

---

## CRITICAL ARCHITECTURAL FLAW

### Issue #0: Dual Independent Agents Cannot Coordinate

**Severity:** CRITICAL - ARCHITECTURAL
**Files:**
- `src/game/agent_steering.py`
- `src/game/agent_speed.py`
- `src/game/training.py`

**Problem:**
The system uses two completely independent DQN agents:
1. Steering agent - outputs steering angle only
2. Speed agent - outputs throttle/brake only

These agents:
- Have separate neural networks
- Have separate replay buffers
- Make independent decisions without knowing the other's action
- Receive separate (potentially conflicting) rewards

**Impact:**
- **Impossible to learn proper racing behavior** - racing requires coordinated steering + speed control
- Example failure modes:
  - Agent steers hard left while accelerating → crash
  - Agent brakes while going straight → loses race
  - No way to learn "slow down before sharp turns"
- Reward attribution problem - if car crashes, which agent is at fault?
- Cannot learn state-action relationships that involve both controls

**Real-world analogy:**
It's like having two drivers in a car - one controls the steering wheel, the other controls the pedals, and they can't communicate. This will never work.

**Correct Architecture:**
Single agent with multi-dimensional action space:
- Input: Game state (screen capture, speed, position)
- Output: (steering, throttle) as a single coordinated action
- Single replay buffer learning the joint action space
- Reward based on combined performance

---

## CRITICAL RUNTIME ERRORS

### Issue #1: Missing Learning Object Initialization

**Severity:** CRITICAL
**Files:**
- `src/game/agent_steering.py` line 85
- `src/game/agent_speed.py` line 85

**Problem:**
```python
def learn(self, action):
    # ... code ...
    self.learning.learn(action)  # ERROR: self.learning never initialized!
```

The `__init__` method never creates `self.learning`, but `learn()` method calls it.

**Impact:**
```
AttributeError: 'SteeringAgent' object has no attribute 'learning'
```
Training crashes immediately on first learn call.

**Fix:**
Add to `__init__`:
```python
self.learning = SteeringLearning(
    model=self.model,
    memory=self.memory,
    expected_shape=self.expected_shape,
    gamma=self.gamma,
    eps=self.eps,
    eps_dec=self.eps_dec,
    eps_min=self.eps_min
)
```

---

### Issue #2: Broken DQN Q-Value Update Logic

**Severity:** CRITICAL - BREAKS LEARNING
**File:** `src/game/learning.py` line 71

**Problem:**
```python
# Current (WRONG):
Q_target = reward + self.gamma * next_action_values * ~done

# This creates a scalar/vector that doesn't match Q_values shape
```

In DQN, you need to:
1. Copy current Q-values
2. Update ONLY the Q-value for the action that was taken
3. Use the updated Q-target for training

**Impact:**
- The steering agent cannot learn at all
- Q-value updates don't follow Bellman equation properly
- Training will fail to converge or diverge

**Fix:**
```python
Q_target = action_values.copy()
batch_index = np.arange(len(action))
Q_target[batch_index, action] = reward + self.gamma * next_action_values * ~done
```

---

### Issue #3: Hard-Coded Batch Size Causes IndexError

**Severity:** CRITICAL
**File:** `src/game/learning.py` line 121

**Problem:**
```python
batch_index = np.arange(32, dtype=int)  # Assumes batch size = 32
Q_target[batch_index, action] = reward + ...
```

After filtering invalid states (lines 100-110), the batch size may be less than 32, causing:
```
IndexError: index 31 is out of bounds for axis 0 with size 15
```

**Impact:**
Training crashes randomly when filtered batch < 32 samples.

**Fix:**
```python
batch_index = np.arange(len(action), dtype=int)
```

---

### Issue #4: Episodes Never Terminate

**Severity:** CRITICAL - INFINITE LOOPS
**File:** `src/game/game_env.py` lines 30-32

**Problem:**
```python
def is_done(self, img):
    """Check if the game is over (e.g., car crash). Placeholder for now."""
    return False  # Always False!
```

**Impact:**
- Training loops run forever: `while not done:` never exits
- Episodes never complete
- Agent never gets episode-end learning signal
- No concept of terminal states in RL

**Fix Options:**
1. Implement crash detection (detect crash screen/UI elements)
2. Add maximum step counter:
```python
def is_done(self, img):
    self.step_count += 1
    if self.step_count > self.max_steps:
        return True
    # TODO: Add crash detection
    return self._detect_crash(img)
```

---

### Issue #5: Function Signature Mismatch

**Severity:** CRITICAL
**File:** `src/game/training.py` lines 88, 96

**Problem:**
```python
# Function definition:
def train_steering(agent, num_episodes, screen=None, detection_info=None):
    ...

# Function call:
train_steering(agent, screen, detection_info, episodes)
# Arguments in wrong order! 'screen' goes into 'num_episodes' parameter
```

**Impact:**
```
TypeError: 'numpy.ndarray' object cannot be interpreted as an integer
```

**Fix:**
```python
train_steering(agent, episodes, screen, detection_info)
# Or use keyword arguments:
train_steering(agent, num_episodes=episodes, screen=screen, detection_info=detection_info)
```

---

### Issue #6: Array Dimension Swap in Boundary Check

**Severity:** CRITICAL
**Files:**
- `src/game/game_env.py` line 131
- `src/game/game_capture.py` line 131
- `src/game/detection.py` line 31

**Problem:**
```python
position_red = loc_red[1][0]    # This is WIDTH (x-coordinate)
position_blue = loc_blue[1][0]  # This is WIDTH (x-coordinate)

# Wrong dimension check:
if position_red >= img.shape[0] or position_blue >= img.shape[1]:
#                     ↑ HEIGHT           ↑ WIDTH
```

Should be:
```python
if position_red >= img.shape[1] or position_blue >= img.shape[1]:
#                     ↑ WIDTH            ↑ WIDTH
```

**Impact:**
- Incorrect boundary validation
- Valid positions rejected incorrectly
- Potential out-of-bounds array access
- Silent failures in chevron detection

**Fix:**
```python
if position_red >= img.shape[1] or position_blue >= img.shape[1]:
    return None, None, None
```

---

### Issue #7: Both Colors Sampled from Same Location

**Severity:** CRITICAL - LOGIC ERROR
**Files:**
- `src/game/game_capture.py` lines 134-135
- `src/game/detection.py` lines 35-36

**Problem:**
```python
colour_red = img[position_red, position_blue]
colour_blue = img[position_red, position_blue]  # SAME COORDINATES!
```

Both colors are sampled from the identical pixel, making `colour_red == colour_blue` always.

**Impact:**
- Speed-up color detection is meaningless
- `is_speed_up_colour` calculation is always wrong
- Speed reward function gets garbage input

**Fix:**
```python
colour_red = img[position_red, position_red]
colour_blue = img[position_blue, position_blue]
```

Or better - sample from center of detected template regions.

---

## HIGH PRIORITY ISSUES

### Issue #8: Learning Object Recreated Every Call

**Severity:** HIGH - PERFORMANCE & CORRECTNESS
**File:** `src/game/agents.py` lines 63-75, 93-105

**Problem:**
```python
def learn(self, action):
    self.learning = SteeringLearning(...)  # NEW OBJECT EVERY TIME!
    self.learning.learn(action)
```

**Impact:**
- Epsilon resets to initial value (1.0) every learning step
- Learning state is not persistent
- Massive performance overhead (model recreation)
- Q-network training disrupted
- No learning progress retained

**Fix:**
Initialize once in `__init__`, reuse throughout:
```python
def __init__(self, ...):
    self.learning = SteeringLearning(...)

def learn(self, action):
    self.learning.learn(action)
```

---

### Issue #9: Action Type Mismatch - Discrete vs Continuous

**Severity:** HIGH
**File:** `src/game/agent_steering.py` lines 45, 67-82

**Problem:**
```python
# Define discrete action space:
self.action_space = [i for i in range(n_actions)]  # [0, 1]

# But return continuous actions:
if np.random.random() < self.eps:
    action = np.random.uniform(-1, 1)  # Continuous in [-1, 1]!
```

**Impact:**
- Inconsistent action representation
- DQN expects discrete actions for Q-table indexing
- Controller receives wrong action format
- Learning algorithm confusion

**Fix Option 1 - Discrete:**
```python
if np.random.random() < self.eps:
    action = np.random.choice(self.action_space)
else:
    action = np.argmax(self.model.predict(state))
# Then map discrete action to continuous controller value
```

**Fix Option 2 - Continuous:**
Use DDPG or SAC instead of DQN for continuous control.

---

### Issue #10: Incorrect Learn Method Interface

**Severity:** HIGH
**File:** `src/game/training.py` lines 41, 74

**Problem:**
```python
agent.learn(action)  # Only passes current action
```

But DQN learning needs entire experience tuple: `(state, action, reward, next_state, done)`

**Impact:**
- Learning architecture doesn't follow standard DQN
- Experience replay not properly implemented
- Agent can't learn from past experiences

**Fix:**
```python
# Store experience:
agent.store_experience(state, action, reward, next_state, done)

# Learn from replay buffer:
if len(agent.memory) > batch_size:
    agent.learn()  # Samples minibatch internally
```

---

### Issue #11: Detection Return Value Inconsistency

**Severity:** HIGH
**Files:**
- `src/game/detection.py` line 51
- `src/game/game_capture.py` line 137

**Problem:**
```python
# detection.py returns 4 values:
return position_red, position_blue, is_speed_up_colour, detected_chevrons

# game_capture.py returns 3 values:
return position_red, position_blue, is_speed_up_colour
```

**Impact:**
```
ValueError: too many values to unpack (expected 3)
```
If wrong module is imported.

**Fix:**
Standardize on 3-value return, make `detected_chevrons` optional parameter.

---

### Issue #12: Thread Race Condition on Shared Game Environment

**Severity:** HIGH
**File:** `src/game/training.py` lines 79-119

**Problem:**
```python
game_env = GameEnvironment()  # Global shared object

# Both threads access simultaneously:
def train_steering_thread():
    agent.game_env.capture()  # Thread 1

def train_speed_thread():
    agent.game_env.capture()  # Thread 2
```

**Impact:**
- Race condition on screen capture
- Corrupted/inconsistent image data
- Training instability
- Undefined behavior

**Fix:**
```python
# Option 1: Thread locks
import threading
capture_lock = threading.Lock()

with capture_lock:
    screen = game_env.capture()

# Option 2: Separate environments per thread
steering_env = GameEnvironment()
speed_env = GameEnvironment()
```

---

## MEDIUM PRIORITY ISSUES

### Issue #13: Duplicate Code - game_env.py vs game_capture.py

**Severity:** MEDIUM
**Files:** `src/game/game_env.py`, `src/game/game_capture.py`

**Problem:**
Both files define `GameEnvironment` class with nearly identical methods:
- `get_chevron_info()`
- `speed_reward()`
- `steering_reward()`
- `capture()`

**Impact:**
- Code maintenance burden
- Bug fixes needed in two places
- Diverging implementations
- Confusion about which is canonical

**Fix:**
Delete `game_capture.py` or merge into single source of truth.

---

### Issue #14: Invalid Empty Array Check

**Severity:** MEDIUM
**Files:**
- `src/game/detection.py` line 24
- `src/game/game_capture.py` line 121

**Problem:**
```python
if not loc[0].any() and not loc[1].any():
    return None, None, None
```

`.any()` returns True if ANY element is True. For checking empty arrays:
```python
if loc[0].size == 0 or loc[1].size == 0:
    return None, None, None
```

**Impact:**
- Valid detections may be incorrectly rejected
- Template matching fails silently

---

### Issue #15: Redundant State Preprocessing

**Severity:** MEDIUM - PERFORMANCE
**File:** `src/game/learning.py` lines 36, 44-48

**Problem:**
```python
state = np.array([self.preprocess_input_data(s) for s in state])  # Line 36
# ... validation ...
state = [s for s in state if s.shape == self.expected_shape]  # Line 44
state = np.array([self.preprocess_input_data(s) for s in state])  # Line 45 - REDUNDANT!
```

**Impact:**
- Preprocessing applied twice per training step
- ~50% performance overhead in data pipeline

**Fix:**
Apply preprocessing once before validation.

---

### Issue #16: Epsilon Not Persisted Across Learning Objects

**Severity:** MEDIUM
**File:** `src/game/learning.py` lines 75-77, 129-131

**Problem:**
Epsilon stored in learning object, which is recreated every call (see Issue #8).

**Impact:**
- Epsilon never decays across training
- Agent doesn't transition from exploration to exploitation
- Always random actions

**Fix:**
Store epsilon in agent, pass by reference to learning.

---

### Issue #17: Epsilon Decays Too Fast

**Severity:** MEDIUM - RL HYPERPARAMETER
**File:** `src/game/learning.py` lines 75-77

**Problem:**
```python
if self.eps > self.eps_min:
    self.eps *= self.eps_dec  # 0.995 per learning step
```

With 32-sample minibatch learning every step:
- 1000 steps/episode × 100 episodes = 100,000 learning calls
- Epsilon: 1.0 × 0.995^100000 ≈ 0 (effectively zero very quickly)

**Impact:**
- Exploration stops too early
- Agent gets stuck in local optima
- Never explores full state space

**Fix:**
Decay per episode, not per learning step:
```python
# In training loop after episode ends:
agent.decay_epsilon()
```

---

### Issue #18: No Batch Size Validation After Filtering

**Severity:** MEDIUM
**File:** `src/game/learning.py` lines 84-85, 111-112

**Problem:**
```python
if len(self.memory) < 32:
    return
# ... sample 32 experiences ...
# ... filter by shape ...
# No check if filtered batch still has enough samples
```

**Impact:**
- Training on very small batches (e.g., 2-3 samples)
- High variance in gradient updates
- Training instability

**Fix:**
```python
if len(valid_states) < min_batch_size:
    return  # Skip this training step
```

---

### Issue #19: Missing Action Range Clipping

**Severity:** MEDIUM
**File:** `src/game/controller.py` lines 8, 15, 18

**Problem:**
```python
x_value = int(action * 32767)  # Assumes action ∈ [-1, 1]
self.controller.right_trigger(value=int(action * 255))  # Assumes action ∈ [0, 1]
```

No validation that action is in expected range.

**Impact:**
- Out-of-range actions cause overflow/underflow
- Controller receives invalid values
- Unpredictable game control

**Fix:**
```python
x_value = int(np.clip(action, -1, 1) * 32767)
self.controller.right_trigger(value=int(np.clip(action, 0, 1) * 255))
```

---

### Issue #20: Steering Reward Doesn't Penalize Action Magnitude

**Severity:** MEDIUM - REWARD DESIGN
**File:** `src/game/rewards.py`

**Problem:**
Speed reward includes action penalty (line 25):
```python
reward -= abs(action) * 0.1  # Penalize large throttle changes
```

But steering reward has no such penalty → encourages jerky steering.

**Impact:**
- Agent learns to make large, sudden steering movements
- Unstable driving behavior
- Doesn't learn smooth control

**Fix:**
```python
def steering_reward(self, position_red, position_blue, action):
    reward = ...
    reward -= abs(action) * 0.05  # Penalize large steering angles
    return reward
```

---

### Issue #21: Memory Not Persisted Between Training Sessions

**Severity:** MEDIUM
**File:** `src/game/agents.py` line 16

**Problem:**
```python
self.memory = deque(maxlen=mem_size)  # Fresh memory every initialization
```

**Impact:**
- No continuity between training runs
- Can't resume training from checkpoint
- Loses valuable experiences

**Fix:**
Implement save/load for replay buffer:
```python
def save_memory(self, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(list(self.memory), f)

def load_memory(self, filepath):
    with open(filepath, 'rb') as f:
        self.memory = deque(pickle.load(f), maxlen=self.memory.maxlen)
```

---

### Issue #22: No Resource Cleanup for Screen Capture

**Severity:** MEDIUM - RESOURCE LEAK
**Files:** `src/game/image_capture.py`, `src/game/game_capture.py`

**Problem:**
`mss` screen capture objects are created but never explicitly closed.

**Impact:**
- Resource leaks over long training runs
- Accumulating memory usage
- Potential handle exhaustion

**Fix:**
```python
class GameEnvironment:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'sct'):
            self.sct.close()
```

Use with context manager:
```python
with GameEnvironment() as env:
    # training loop
```

---

### Issue #23: No Error Handling for Window Detection Failure

**Severity:** MEDIUM
**File:** `src/game/main.py`

**Problem:**
If game window isn't found, exception is raised with no recovery:
```python
game_env = GameEnvironment()  # Raises if window not found
```

**Impact:**
- Program crashes immediately
- No user-friendly error message
- No retry logic

**Fix:**
```python
max_retries = 5
for attempt in range(max_retries):
    try:
        game_env = GameEnvironment()
        break
    except WindowNotFoundError as e:
        print(f"Attempt {attempt+1}/{max_retries}: Game window not found")
        time.sleep(2)
else:
    print("ERROR: Could not find game window after 5 attempts")
    sys.exit(1)
```

---

### Issue #24: No Image Channel Validation

**Severity:** MEDIUM
**File:** `src/game/image_processing.py` lines 14-22

**Problem:**
```python
if len(img.shape) == 3:
    for i in range(3):  # Assumes exactly 3 channels
        features.append(np.mean(img[:,:,i]))
```

If image has 4 channels (RGBA), this silently ignores the alpha channel.

**Impact:**
- Inconsistent feature extraction
- Wrong input to neural network
- Training on incorrect state representation

**Fix:**
```python
if len(img.shape) == 3:
    # Convert to RGB if needed
    if img.shape[2] == 4:
        img = img[:,:,:3]  # Drop alpha
    for i in range(img.shape[2]):
        features.append(np.mean(img[:,:,i]))
```

---

### Issue #25: Learning Rate May Be Too Small

**Severity:** MEDIUM - HYPERPARAMETER
**File:** `src/game/model.py` line 11

**Problem:**
```python
model.compile(optimizer=Adam(learning_rate=0.001), ...)
```

For RL, 0.001 is often too small, especially early in training.

**Impact:**
- Very slow convergence
- Training takes prohibitively long
- May not converge at all in reasonable time

**Fix:**
```python
model.compile(optimizer=Adam(learning_rate=0.01), ...)
# Or make it configurable
```

---

## MINOR ISSUES

### Issue #26: Missing Space in Assignment

**Severity:** MINOR - CODE STYLE
**Files:**
- `src/game/agent_steering.py` line 36
- `src/game/agent_speed.py` line 50

**Problem:**
```python
self.expected_shape =game_env.expected_shape  # Missing space before =
```

**Fix:**
```python
self.expected_shape = game_env.expected_shape
```

---

### Issue #27: Inconsistent Model Save Path Handling

**Severity:** MINOR
**Files:** Various

**Problem:**
Model save/load paths are hardcoded in some places, configurable in others.

**Fix:**
Centralize path configuration.

---

## SUMMARY STATISTICS

| Category | Count |
|----------|-------|
| **Architectural** | 1 |
| **Critical** | 7 |
| **High Priority** | 5 |
| **Medium Priority** | 13 |
| **Minor** | 2 |
| **TOTAL** | 28 |

### Affected Components

| Component | Issues |
|-----------|--------|
| Agent Architecture | 8 |
| Learning Algorithm | 6 |
| Game Environment | 5 |
| Detection/Capture | 4 |
| Training Loop | 3 |
| Rewards | 2 |

---

## RECOMMENDED ACTION PLAN

### Phase 1: Architecture Redesign (CRITICAL)

**Must complete before fixing other bugs**

1. **Design unified agent architecture**
   - Single agent class with multi-dimensional action output
   - Combined action space: `(steering, throttle, brake)` or `(steering, speed)`
   - Single replay buffer
   - Single Q-network with multi-output head

2. **Update neural network model**
   - Output layer: 2-3 continuous values or discretized action combinations
   - Consider using DDPG/TD3 for continuous control or discretize action space

3. **Redesign reward function**
   - Single reward considering both steering and speed
   - Encourage coordinated behavior (e.g., slow down before turns)

### Phase 2: Critical Bug Fixes

Fix in this order:

1. ✅ Implement unified agent (Issue #0)
2. ✅ Fix DQN Q-value update (Issue #2)
3. ✅ Fix episode termination (Issue #4)
4. ✅ Fix array indexing bugs (Issues #6, #7)
5. ✅ Fix learning object initialization (Issues #1, #8)
6. ✅ Fix function signatures (Issue #5)

### Phase 3: High Priority Fixes

7. Fix action space consistency (Issue #9)
8. Fix learn method interface (Issue #10)
9. Fix thread safety (Issue #12)
10. Standardize return values (Issue #11)

### Phase 4: Medium Priority Improvements

11. Remove duplicate code (Issue #13)
12. Fix epsilon decay (Issues #16, #17)
13. Add proper validation (Issues #14, #18, #19, #24)
14. Improve reward design (Issue #20)
15. Add resource cleanup (Issues #22, #23)
16. Optimize performance (Issues #15, #25)

### Phase 5: Testing & Validation

- Unit tests for each component
- Integration test for training loop
- Validate agent can learn simple tasks
- Tune hyperparameters

---

## TECHNICAL NOTES

### Recommended Agent Architecture

```python
class RacingAgent:
    """Unified agent controlling both steering and speed"""

    def __init__(self, state_shape, learning_rate=0.01):
        # Single model with multi-dimensional output
        # Option 1: Discrete action space (combination of steering × speed)
        # Option 2: Continuous action space (use DDPG/TD3)

        self.model = self._build_model(state_shape)
        self.target_model = self._build_model(state_shape)
        self.memory = deque(maxlen=10000)

    def _build_model(self, state_shape):
        # Input: game state
        # Output: Q-values for each (steering, speed) combination
        # OR: (steering, speed) continuous values with actor-critic
        pass

    def choose_action(self, state):
        """Returns (steering, speed) as coordinated action"""
        if np.random.random() < self.epsilon:
            # Random exploration
            steering = np.random.uniform(-1, 1)
            speed = np.random.uniform(0, 1)
        else:
            # Greedy exploitation
            q_values = self.model.predict(state)
            action_idx = np.argmax(q_values)
            steering, speed = self._decode_action(action_idx)

        return steering, speed

    def store_experience(self, state, action, reward, next_state, done):
        """action is tuple (steering, speed)"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Learn from minibatch with proper DQN update"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        # ... proper Q-learning update ...
```

### Recommended Reward Function

```python
def compute_reward(self, state, action, next_state):
    """Unified reward considering both steering and speed"""
    steering, speed = action

    # Position reward (stay on track)
    position_reward = self._position_reward(next_state)

    # Speed reward (go fast on straights)
    speed_reward = speed * 0.5

    # Smoothness penalty (discourage jerky control)
    smoothness_penalty = abs(steering) * 0.1 + abs(speed - self.prev_speed) * 0.1

    # Crash penalty
    crash_penalty = -100 if self.is_crashed(next_state) else 0

    total_reward = position_reward + speed_reward - smoothness_penalty + crash_penalty

    self.prev_speed = speed
    return total_reward
```

---

## CONCLUSION

The ForzaAIMasters project has a solid foundation but requires significant refactoring to function correctly. The most critical issue is the architectural flaw of using two independent agents, which fundamentally cannot learn coordinated racing behavior.

**Estimated Effort:**
- Architecture redesign: 4-6 hours
- Bug fixes: 3-4 hours
- Testing & validation: 2-3 hours
- **Total: 9-13 hours** of development work

**Priority:** Address architectural flaw first, then systematic bug fixes, then optimization.

---

*End of Report*
