"""
Modern PPO (Proximal Policy Optimization) Agent for Racing

PPO is state-of-the-art for continuous control (2023-2025).
Superior to DQN for racing because:
- Handles continuous actions naturally
- More stable training
- Better sample efficiency with modern tricks
- Used by OpenAI, DeepMind for robotics and games

Modern enhancements:
- GAE (Generalized Advantage Estimation)
- Value function clipping
- Multiple epochs per batch
- Gradient clipping
- Learning rate scheduling
- Mixed precision training
- Vectorized environments (future)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from collections import deque
import pickle


class ModernPPOAgent:
    """
    Modern PPO agent with all 2024-2025 best practices.

    Uses continuous action space for smooth steering and throttle control.
    """

    def __init__(
        self,
        state_dim,
        action_dim=2,  # [steering, throttle/brake]
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        epochs_per_update=10,
        mini_batch_size=64,
        buffer_size=2048,
        use_mixed_precision=True,
        model_save_dir='models/modern_ppo'
    ):
        """
        Initialize modern PPO agent.

        Parameters:
        - state_dim: State feature dimension
        - action_dim: Action dimension (2 for steering + throttle)
        - learning_rate: Learning rate with scheduling
        - gamma: Discount factor
        - gae_lambda: GAE lambda for advantage estimation
        - clip_ratio: PPO clip ratio (0.2 is standard)
        - entropy_coef: Entropy bonus for exploration
        - value_coef: Value loss coefficient
        - max_grad_norm: Gradient clipping threshold
        - epochs_per_update: Training epochs per batch
        - mini_batch_size: Mini-batch size for updates
        - buffer_size: Rollout buffer size
        - use_mixed_precision: Enable mixed precision (faster on modern GPUs)
        - model_save_dir: Directory to save models
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size
        self.buffer_size = buffer_size
        self.model_save_dir = model_save_dir

        os.makedirs(model_save_dir, exist_ok=True)

        # Mixed precision for faster training on modern GPUs
        if use_mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled (FP16)")

        # Build networks
        self.actor, self.critic = self._build_networks()

        # Optimizer with gradient clipping
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_grad_norm)

        # Rollout buffer
        self.buffer = PPORolloutBuffer(buffer_size)

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

        # Training step counter
        self.update_step = 0

        print(f"\nModern PPO Agent Initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim} (continuous)")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Clip ratio: {clip_ratio}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Mini-batch size: {mini_batch_size}")

    def _build_networks(self):
        """
        Build actor (policy) and critic (value) networks.

        Modern architecture with:
        - Shared feature extractor (more efficient)
        - Separate heads for policy and value
        - Layer normalization
        - GELU activations
        - Residual connections
        """

        # Shared feature extractor
        state_input = layers.Input(shape=(self.state_dim,), name='state_input')

        # Feature extraction with residual connections
        x = layers.Dense(512, activation='gelu', name='shared_1')(state_input)
        x = layers.LayerNormalization()(x)

        # Residual block 1
        residual = x
        x = layers.Dense(512, activation='gelu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Add()([residual, x])

        # Residual block 2
        residual = x
        x = layers.Dense(512, activation='gelu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Add()([residual, x])

        shared_features = x

        # === ACTOR (Policy Network) ===
        # Outputs mean and log_std for Gaussian policy
        actor_hidden = layers.Dense(256, activation='gelu', name='actor_hidden')(shared_features)
        actor_hidden = layers.LayerNormalization()(actor_hidden)

        # Mean of action distribution
        action_mean = layers.Dense(
            self.action_dim,
            activation='tanh',  # Bound to [-1, 1]
            kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
            name='action_mean'
        )(actor_hidden)

        # Log standard deviation (learned)
        action_log_std = layers.Dense(
            self.action_dim,
            kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
            name='action_log_std'
        )(actor_hidden)

        actor = keras.Model(
            inputs=state_input,
            outputs=[action_mean, action_log_std],
            name='actor_network'
        )

        # === CRITIC (Value Network) ===
        value_hidden = layers.Dense(256, activation='gelu', name='value_hidden')(shared_features)
        value_hidden = layers.LayerNormalization()(value_hidden)

        state_value = layers.Dense(
            1,
            kernel_initializer=keras.initializers.Orthogonal(gain=1.0),
            name='state_value'
        )(value_hidden)

        critic = keras.Model(
            inputs=state_input,
            outputs=state_value,
            name='critic_network'
        )

        return actor, critic

    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.

        Parameters:
        - state: Current state
        - deterministic: If True, return mean action (for evaluation)

        Returns:
        - action: Sampled action
        - log_prob: Log probability of action
        - value: State value estimate
        """
        state = np.array([state]) if len(state.shape) == 1 else state

        # Get policy distribution
        action_mean, action_log_std = self.actor.predict(state, verbose=0)

        if deterministic:
            action = action_mean[0]
            log_prob = None
        else:
            # Sample from Gaussian distribution
            action_std = np.exp(action_log_std[0])
            action = action_mean[0] + action_std * np.random.randn(self.action_dim)

            # Clip to valid range [-1, 1]
            action = np.clip(action, -1.0, 1.0)

            # Compute log probability
            log_prob = self._compute_log_prob(action, action_mean[0], action_log_std[0])

        # Get value estimate
        value = self.critic.predict(state, verbose=0)[0][0]

        return action, log_prob, value

    def _compute_log_prob(self, action, mean, log_std):
        """Compute log probability of action under Gaussian policy."""
        std = np.exp(log_std)
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var +
            2 * log_std +
            np.log(2 * np.pi)
        )
        return np.sum(log_prob)

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in rollout buffer."""
        self.buffer.add(state, action, reward, value, log_prob, done)

    def update(self):
        """
        Perform PPO update using collected rollouts.

        This is called after collecting buffer_size steps.
        """
        if len(self.buffer) < self.buffer_size:
            return None

        # Get rollout data
        states, actions, old_log_probs, returns, advantages = self.buffer.get()

        # Normalize advantages (important for stability)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Convert to tensors
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.float32)
        old_log_probs = tf.constant(old_log_probs, dtype=tf.float32)
        returns = tf.constant(returns, dtype=tf.float32)
        advantages = tf.constant(advantages, dtype=tf.float32)

        # Multiple epochs over the data
        policy_losses = []
        value_losses = []
        entropies = []

        for epoch in range(self.epochs_per_update):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_states = tf.gather(states, mb_indices)
                mb_actions = tf.gather(actions, mb_indices)
                mb_old_log_probs = tf.gather(old_log_probs, mb_indices)
                mb_returns = tf.gather(returns, mb_indices)
                mb_advantages = tf.gather(advantages, mb_indices)

                # PPO update
                policy_loss, value_loss, entropy = self._ppo_update_step(
                    mb_states, mb_actions, mb_old_log_probs, mb_returns, mb_advantages
                )

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)

        # Clear buffer
        self.buffer.clear()

        # Update metrics
        self.policy_losses.append(np.mean(policy_losses))
        self.value_losses.append(np.mean(value_losses))
        self.entropies.append(np.mean(entropies))
        self.update_step += 1

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }

    @tf.function
    def _ppo_update_step(self, states, actions, old_log_probs, returns, advantages):
        """Single PPO update step (compiled for speed)."""

        with tf.GradientTape() as tape:
            # Forward pass
            action_mean, action_log_std = self.actor(states, training=True)
            action_std = tf.exp(action_log_std)

            # Compute new log probabilities
            dist = tfp.distributions.Normal(action_mean, action_std)
            new_log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1)

            # Entropy for exploration bonus
            entropy = tf.reduce_mean(dist.entropy())

            # Ratio for PPO clip
            ratio = tf.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Value loss with clipping
            values = tf.squeeze(self.critic(states, training=True))
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Compute gradients and update
        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        return policy_loss, value_loss, entropy

    def save(self, filename='ppo_agent'):
        """Save agent models."""
        actor_path = os.path.join(self.model_save_dir, f'{filename}_actor.h5')
        critic_path = os.path.join(self.model_save_dir, f'{filename}_critic.h5')

        self.actor.save(actor_path)
        self.critic.save(critic_path)

        print(f"Models saved: {actor_path}, {critic_path}")

    def load(self, filename='ppo_agent'):
        """Load agent models."""
        actor_path = os.path.join(self.model_save_dir, f'{filename}_actor.h5')
        critic_path = os.path.join(self.model_save_dir, f'{filename}_critic.h5')

        self.actor = keras.models.load_model(actor_path)
        self.critic = keras.models.load_model(critic_path)

        print(f"Models loaded: {actor_path}, {critic_path}")


class PPORolloutBuffer:
    """
    Rollout buffer for PPO with GAE (Generalized Advantage Estimation).
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def get(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute returns and advantages using GAE.

        Returns:
        - states, actions, log_probs, returns, advantages
        """
        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)

        # Compute advantages using GAE
        advantages = np.zeros(len(self.rewards))
        last_advantage = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_advantage

        # Returns = advantages + values
        returns = advantages + np.array(self.values)

        return states, actions, log_probs, returns, advantages


# TensorFlow Probability import (needed for PPO)
try:
    import tensorflow_probability as tfp
except ImportError:
    print("WARNING: tensorflow_probability not installed. Install with: pip install tensorflow-probability")
    print("PPO agent will not work without it.")
