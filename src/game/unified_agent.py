"""
Unified Racing Agent - Controls both steering and speed simultaneously

This agent uses a single neural network to output coordinated actions for both
steering and throttle/brake control. It implements Deep Q-Network (DQN) with
experience replay and target networks for stable learning.

Key improvements over dual-agent system:
- Coordinated action space (steering + speed together)
- Proper Q-learning with Bellman equation
- Progressive learning from novice to pro
- Curriculum learning support
- Robust error handling and state validation
"""

import os
import numpy as np
import pickle
from collections import deque
from tensorflow.keras.models import load_model
from controller import GameController


class UnifiedRacingAgent:
    """
    Unified agent that controls both steering and speed/throttle.

    Action Space:
    - Discrete grid combining steering x speed decisions
    - steering_actions: [-1.0, -0.5, 0.0, 0.5, 1.0] (5 discrete values)
    - speed_actions: [-1.0, -0.5, 0.0, 0.5, 1.0] (5 discrete values)
    - Total: 5 x 5 = 25 discrete action combinations

    This allows coordinated decisions like:
    - "Turn hard left while braking" = (-1.0, -1.0)
    - "Go straight and accelerate" = (0.0, 1.0)
    - "Slight right while maintaining speed" = (0.5, 0.0)
    """

    # Define discrete action spaces
    STEERING_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    SPEED_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    def __init__(
        self,
        game_env,
        input_dims,
        mem_size=50000,
        batch_size=64,
        eps=1.0,
        eps_min=0.01,
        eps_dec=0.995,
        gamma=0.99,
        learning_rate=0.001,
        target_update_freq=1000,
        model_save_dir='models/unified_agent'
    ):
        """
        Initialize the Unified Racing Agent.

        Parameters:
        - game_env: Game environment instance
        - input_dims (int): Number of input features
        - mem_size (int): Maximum replay buffer size
        - batch_size (int): Minibatch size for learning
        - eps (float): Initial exploration rate
        - eps_min (float): Minimum exploration rate
        - eps_dec (float): Exploration decay rate (per episode)
        - gamma (float): Discount factor for future rewards
        - learning_rate (float): Neural network learning rate
        - target_update_freq (int): Steps between target network updates
        - model_save_dir (str): Directory to save model checkpoints
        """
        self.game_env = game_env
        self.input_dims = input_dims
        self.expected_shape = game_env.expected_shape

        # Action space setup
        self.n_steering_actions = len(self.STEERING_VALUES)
        self.n_speed_actions = len(self.SPEED_VALUES)
        self.n_actions = self.n_steering_actions * self.n_speed_actions  # 25 total actions

        # Hyperparameters
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        # Experience replay buffer
        self.memory = deque(maxlen=mem_size)

        # Learning step counter
        self.learn_step = 0
        self.episode_count = 0

        # Model save directory
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)

        # Build Q-networks (evaluation and target)
        from unified_model import build_unified_model
        self.q_eval = build_unified_model(input_dims, self.n_actions, learning_rate)
        self.q_target = build_unified_model(input_dims, self.n_actions, learning_rate)
        self.update_target_network()

        # Game controller
        self.controller = GameController()

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')

        print(f"Unified Racing Agent initialized:")
        print(f"  - Action space: {self.n_steering_actions} steering × {self.n_speed_actions} speed = {self.n_actions} total")
        print(f"  - Input dimensions: {input_dims}")
        print(f"  - Replay buffer size: {mem_size}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Initial epsilon: {eps}")

    def decode_action(self, action_idx):
        """
        Convert discrete action index to (steering, speed) tuple.

        Parameters:
        - action_idx (int): Action index in range [0, n_actions-1]

        Returns:
        - (steering, speed) tuple of continuous values
        """
        steering_idx = action_idx // self.n_speed_actions
        speed_idx = action_idx % self.n_speed_actions

        steering = self.STEERING_VALUES[steering_idx]
        speed = self.SPEED_VALUES[speed_idx]

        return steering, speed

    def encode_action(self, steering, speed):
        """
        Convert (steering, speed) continuous values to discrete action index.

        Parameters:
        - steering (float): Steering value
        - speed (float): Speed value

        Returns:
        - action_idx (int): Discrete action index
        """
        # Find closest discrete values
        steering_idx = np.argmin(np.abs(self.STEERING_VALUES - steering))
        speed_idx = np.argmin(np.abs(self.SPEED_VALUES - speed))

        return steering_idx * self.n_speed_actions + speed_idx

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.

        Parameters:
        - state (array): Current state observation

        Returns:
        - action_idx (int): Chosen action index
        - (steering, speed) tuple: Decoded action values
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.eps:
            # Random exploration
            action_idx = np.random.randint(0, self.n_actions)
        else:
            # Greedy exploitation
            state_input = np.array([state])
            q_values = self.q_eval.predict(state_input, verbose=0)[0]
            action_idx = np.argmax(q_values)

        # Decode to steering and speed values
        steering, speed = self.decode_action(action_idx)

        return action_idx, (steering, speed)

    def store_transition(self, state, action_idx, reward, next_state, done):
        """
        Store experience in replay buffer.

        Parameters:
        - state (array): Current state
        - action_idx (int): Action taken
        - reward (float): Reward received
        - next_state (array): Next state
        - done (bool): Episode termination flag
        """
        self.memory.append((state, action_idx, reward, next_state, done))

    def learn(self):
        """
        Perform one learning step using experience replay and DQN algorithm.

        This implements the proper Deep Q-Learning update:
        1. Sample random minibatch from replay buffer
        2. Compute target Q-values using Bellman equation
        3. Train Q-network to minimize TD error
        4. Periodically update target network
        """
        # Check if enough experiences in memory
        if len(self.memory) < self.batch_size:
            return None

        # Sample random minibatch
        import random
        minibatch = random.sample(self.memory, self.batch_size)

        # Unpack minibatch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience

            # Validate state shapes
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)

            # Only add if shapes are correct
            if state.shape == (self.input_dims,) and next_state.shape == (self.input_dims,):
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

        # Check if we have enough valid experiences after filtering
        if len(states) < self.batch_size // 2:  # At least half of batch should be valid
            return None

        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=bool)

        # Current Q-values
        current_q_values = self.q_eval.predict(states, verbose=0)

        # Next Q-values from target network
        next_q_values = self.q_target.predict(next_states, verbose=0)

        # Compute target Q-values using Bellman equation
        # Q(s,a) = r + γ * max_a' Q(s', a') if not done
        # Q(s,a) = r if done
        target_q_values = current_q_values.copy()

        batch_indices = np.arange(len(actions))

        # Update Q-values for actions taken
        target_q_values[batch_indices, actions] = rewards + \
            self.gamma * np.max(next_q_values, axis=1) * ~dones

        # Train Q-network
        loss = self.q_eval.train_on_batch(states, target_q_values)

        # Update target network periodically
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()
            print(f"Target network updated at step {self.learn_step}")

        return loss

    def update_target_network(self):
        """Copy weights from Q-evaluation network to target network."""
        self.q_target.set_weights(self.q_eval.get_weights())

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec
            self.eps = max(self.eps, self.eps_min)

    def simulate_action(self, steering, speed):
        """
        Execute coordinated steering and speed action in game.

        Parameters:
        - steering (float): Steering value in [-1, 1]
        - speed (float): Speed/throttle value in [-1, 1]
                        Positive = accelerate, negative = brake
        """
        # Clip values to valid range
        steering = np.clip(steering, -1.0, 1.0)
        speed = np.clip(speed, -1.0, 1.0)

        # Simulate steering
        x_value = int(steering * 32767)
        self.controller.left_joystick(x_value=x_value, y_value=0)

        # Simulate throttle/brake
        if speed >= 0:  # Accelerate
            self.controller.right_trigger(value=int(speed * 255))
            self.controller.left_trigger(value=0)
        else:  # Brake
            self.controller.left_trigger(value=int(-speed * 255))
            self.controller.right_trigger(value=0)

        self.controller.update()

    def save_model(self, filename=None):
        """
        Save Q-evaluation network.

        Parameters:
        - filename (str): Optional custom filename
        """
        if filename is None:
            filename = f'model_episode_{self.episode_count}.h5'

        filepath = os.path.join(self.model_save_dir, filename)
        self.q_eval.save(filepath)
        print(f"Model saved: {filepath}")

        return filepath

    def load_model(self, filepath):
        """
        Load Q-evaluation network from file.

        Parameters:
        - filepath (str): Path to model file
        """
        if os.path.exists(filepath):
            self.q_eval = load_model(filepath)
            self.update_target_network()
            print(f"Model loaded: {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

    def save_memory(self, filename='replay_buffer.pkl'):
        """
        Save replay buffer to disk for training continuation.

        Parameters:
        - filename (str): Filename for replay buffer
        """
        filepath = os.path.join(self.model_save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory), f)
        print(f"Replay buffer saved: {filepath}")

    def load_memory(self, filename='replay_buffer.pkl'):
        """
        Load replay buffer from disk.

        Parameters:
        - filename (str): Filename for replay buffer
        """
        filepath = os.path.join(self.model_save_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
                self.memory = deque(experiences, maxlen=self.mem_size)
            print(f"Replay buffer loaded: {filepath} ({len(self.memory)} experiences)")
        else:
            print(f"Replay buffer file not found: {filepath}")

    def save_training_state(self, episode):
        """
        Save complete training state (model + memory + metrics).

        Parameters:
        - episode (int): Current episode number
        """
        # Save model
        self.save_model(f'checkpoint_episode_{episode}.h5')

        # Save replay buffer
        self.save_memory(f'replay_buffer_episode_{episode}.pkl')

        # Save training metrics
        metrics = {
            'episode': episode,
            'epsilon': self.eps,
            'learn_step': self.learn_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_reward': self.best_reward
        }

        metrics_file = os.path.join(self.model_save_dir, f'metrics_episode_{episode}.pkl')
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)

        print(f"Training state saved at episode {episode}")

    def get_stats(self):
        """
        Get training statistics.

        Returns:
        - dict: Training statistics
        """
        if len(self.episode_rewards) == 0:
            return {
                'episodes': 0,
                'epsilon': self.eps,
                'learn_steps': self.learn_step,
                'avg_reward': 0,
                'best_reward': self.best_reward,
                'memory_size': len(self.memory)
            }

        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards

        return {
            'episodes': len(self.episode_rewards),
            'epsilon': self.eps,
            'learn_steps': self.learn_step,
            'avg_reward_last_100': np.mean(recent_rewards),
            'best_reward': self.best_reward,
            'memory_size': len(self.memory),
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }
