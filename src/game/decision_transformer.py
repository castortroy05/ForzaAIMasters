"""
Decision Transformer - Treating RL as Sequence Modeling (2021-2024)

VERY CUTTING-EDGE approach to RL!

Instead of value functions or policy gradients, treats RL as a
sequence-to-sequence problem using Transformers.

Paper: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
Chen et al., NeurIPS 2021

Why it's revolutionary:
- No value functions, no policy gradients
- Just predict action given (return-to-go, state, action) history
- Can leverage all the Transformer tricks (attention, pre-training, etc.)
- Offline RL friendly (learn from datasets)
- Can specify desired return at inference time
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque


class DecisionTransformer:
    """
    Decision Transformer for racing.

    Predicts actions autoregressively given:
    - Desired return-to-go
    - State history
    - Previous action history
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_episode_length=1000,
        context_length=20,  # Number of timesteps to condition on
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        """
        Initialize Decision Transformer.

        Parameters:
        - state_dim: State dimension
        - action_dim: Action dimension
        - max_episode_length: Maximum episode length
        - context_length: How many past timesteps to condition on
        - embed_dim: Transformer embedding dimension
        - num_layers: Number of Transformer layers
        - num_heads: Number of attention heads
        - dropout: Dropout rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_length = max_episode_length
        self.context_length = context_length
        self.embed_dim = embed_dim

        # Build model
        self.model = self._build_model(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Experience buffer for training
        self.trajectories = []

        print("Decision Transformer initialized")
        print(f"  Context length: {context_length} timesteps")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Transformer layers: {num_layers}")
        print("  Mode: Sequence modeling (not value-based RL)")

    def _build_model(self, embed_dim, num_layers, num_heads, dropout):
        """
        Build Decision Transformer architecture.

        Input sequence format:
        [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_t, s_t]

        Where:
        - R_t = return-to-go at timestep t
        - s_t = state at timestep t
        - a_t = action at timestep t (to be predicted)
        """

        # Input: sequence of (return-to-go, state, action) tuples
        # Shape: (batch, context_length, 3 * embed_dim)
        # We'll embed each component separately then concatenate

        # Return-to-go embeddings
        rtg_input = layers.Input(shape=(self.context_length,), name='rtg_input')
        rtg_embed = layers.Dense(embed_dim, name='rtg_embedding')(
            layers.Reshape((self.context_length, 1))(rtg_input)
        )

        # State embeddings
        state_input = layers.Input(shape=(self.context_length, self.state_dim), name='state_input')
        state_embed = layers.Dense(embed_dim, name='state_embedding')(state_input)

        # Action embeddings (for previous actions)
        action_input = layers.Input(shape=(self.context_length, self.action_dim), name='action_input')
        action_embed = layers.Dense(embed_dim, name='action_embedding')(action_input)

        # Timestep embeddings (positional encoding)
        timestep_input = layers.Input(shape=(self.context_length,), name='timestep_input', dtype=tf.int32)
        timestep_embed = layers.Embedding(
            input_dim=self.max_episode_length,
            output_dim=embed_dim,
            name='timestep_embedding'
        )(timestep_input)

        # Interleave: [rtg_1, s_1, a_1, rtg_2, s_2, a_2, ...]
        # Shape: (batch, context_length * 3, embed_dim)
        sequence = []
        for i in range(self.context_length):
            sequence.append(rtg_embed[:, i:i+1, :])
            sequence.append(state_embed[:, i:i+1, :])
            sequence.append(action_embed[:, i:i+1, :])

        sequence = layers.Concatenate(axis=1)(sequence)

        # Add positional encodings (expanded to match sequence length)
        # Each timestep appears 3 times (rtg, state, action)
        timestep_embed_expanded = tf.repeat(timestep_embed, repeats=3, axis=1)
        sequence = sequence + timestep_embed_expanded

        # Apply Transformer layers
        for i in range(num_layers):
            # Multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout,
                name=f'attention_{i}'
            )(sequence, sequence)

            # Add & Norm
            sequence = layers.Add()([sequence, attn_output])
            sequence = layers.LayerNormalization(epsilon=1e-6)(sequence)

            # Feed-forward network
            ffn = layers.Dense(embed_dim * 4, activation='gelu')(sequence)
            ffn = layers.Dropout(dropout)(ffn)
            ffn = layers.Dense(embed_dim)(ffn)

            # Add & Norm
            sequence = layers.Add()([sequence, ffn])
            sequence = layers.LayerNormalization(epsilon=1e-6)(sequence)

        # Predict actions
        # Extract state positions (every 3rd token starting from index 1)
        # Shape: (batch, context_length, embed_dim)
        state_positions = tf.gather(sequence, indices=range(1, self.context_length * 3, 3), axis=1)

        # Action prediction head
        action_predictions = layers.Dense(embed_dim, activation='gelu')(state_positions)
        action_predictions = layers.Dense(self.action_dim, activation='tanh')(action_predictions)

        model = keras.Model(
            inputs=[rtg_input, state_input, action_input, timestep_input],
            outputs=action_predictions,
            name='decision_transformer'
        )

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse'
        )

        return model

    def get_action(self, states, actions, rtgs, timesteps, target_return=None):
        """
        Get action using the Decision Transformer.

        Parameters:
        - states: State history (context_length, state_dim)
        - actions: Action history (context_length, action_dim)
        - rtgs: Return-to-go history (context_length,)
        - timesteps: Timestep indices (context_length,)
        - target_return: Desired return (optional, for conditioning)

        Returns:
        - action: Next action to take
        """

        # Ensure we have exactly context_length items
        # If less, pad with zeros
        # If more, take the last context_length items

        states = np.array(states)
        actions = np.array(actions)
        rtgs = np.array(rtgs)
        timesteps = np.array(timesteps)

        if len(states) < self.context_length:
            # Pad with zeros
            pad_length = self.context_length - len(states)
            states = np.concatenate([np.zeros((pad_length, self.state_dim)), states], axis=0)
            actions = np.concatenate([np.zeros((pad_length, self.action_dim)), actions], axis=0)
            rtgs = np.concatenate([np.zeros(pad_length), rtgs], axis=0)
            timesteps = np.concatenate([np.zeros(pad_length, dtype=int), timesteps], axis=0)
        elif len(states) > self.context_length:
            # Take last context_length items
            states = states[-self.context_length:]
            actions = actions[-self.context_length:]
            rtgs = rtgs[-self.context_length:]
            timesteps = timesteps[-self.context_length:]

        # Add batch dimension
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)
        rtgs = np.expand_dims(rtgs, axis=0)
        timesteps = np.expand_dims(timesteps, axis=0)

        # Predict action
        action_predictions = self.model.predict(
            [rtgs, states, actions, timesteps],
            verbose=0
        )

        # Return the last predicted action
        action = action_predictions[0, -1, :]

        return action

    def train_on_trajectories(self, trajectories, epochs=10):
        """
        Train on collected trajectories.

        Parameters:
        - trajectories: List of trajectories
          Each trajectory: [(state, action, reward), ...]
        - epochs: Number of training epochs
        """

        # Process trajectories into training data
        train_data = self._process_trajectories(trajectories)

        if not train_data:
            print("No valid training data")
            return

        rtgs_batch, states_batch, actions_batch, timesteps_batch, target_actions = train_data

        # Train
        history = self.model.fit(
            [rtgs_batch, states_batch, actions_batch, timesteps_batch],
            target_actions,
            epochs=epochs,
            batch_size=64,
            verbose=1
        )

        return history

    def _process_trajectories(self, trajectories):
        """
        Process trajectories into training batches.

        Returns:
        - (rtgs, states, actions, timesteps, target_actions)
        """

        rtgs_batch = []
        states_batch = []
        actions_batch = []
        timesteps_batch = []
        target_actions_batch = []

        for traj in trajectories:
            if len(traj) < 2:
                continue

            # Compute return-to-go for each timestep
            rewards = [r for _, _, r in traj]
            rtgs = []
            rtg = 0
            for r in reversed(rewards):
                rtg = r + rtg
                rtgs.insert(0, rtg)

            # Create training samples
            for i in range(len(traj)):
                # Get context window
                start_idx = max(0, i - self.context_length + 1)
                end_idx = i + 1

                context_states = [s for s, _, _ in traj[start_idx:end_idx]]
                context_actions = [a for _, a, _ in traj[start_idx:end_idx]]
                context_rtgs = rtgs[start_idx:end_idx]
                context_timesteps = list(range(start_idx, end_idx))

                # Pad if necessary
                pad_length = self.context_length - len(context_states)
                if pad_length > 0:
                    context_states = [np.zeros(self.state_dim)] * pad_length + context_states
                    context_actions = [np.zeros(self.action_dim)] * pad_length + context_actions
                    context_rtgs = [0] * pad_length + context_rtgs
                    context_timesteps = list(range(pad_length)) + [t + pad_length for t in context_timesteps]

                # Target is the actual actions taken
                target_actions = context_actions.copy()

                rtgs_batch.append(context_rtgs)
                states_batch.append(context_states)
                actions_batch.append(context_actions)
                timesteps_batch.append(context_timesteps)
                target_actions_batch.append(target_actions)

        if not rtgs_batch:
            return None

        return (
            np.array(rtgs_batch),
            np.array(states_batch),
            np.array(actions_batch),
            np.array(timesteps_batch),
            np.array(target_actions_batch)
        )

    def store_trajectory(self, trajectory):
        """Store a complete trajectory for training."""
        self.trajectories.append(trajectory)

        # Keep only recent trajectories (memory management)
        max_trajectories = 1000
        if len(self.trajectories) > max_trajectories:
            self.trajectories = self.trajectories[-max_trajectories:]


class OnlineDecisionTransformer(DecisionTransformer):
    """
    Online Decision Transformer variant.

    Combines Decision Transformer with online learning.
    Can learn from online experience, not just offline datasets.
    """

    def __init__(self, *args, update_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_frequency = update_frequency
        self.steps_since_update = 0

        print("Online Decision Transformer")
        print(f"  Updates every {update_frequency} episodes")

    def maybe_update(self):
        """Maybe perform a training update."""
        self.steps_since_update += 1

        if self.steps_since_update >= self.update_frequency:
            if len(self.trajectories) > 10:
                print(f"\nTraining on {len(self.trajectories)} trajectories...")
                self.train_on_trajectories(self.trajectories, epochs=5)
                self.steps_since_update = 0
