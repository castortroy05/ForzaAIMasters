"""
Neural Network Model for Unified Racing Agent

This model outputs Q-values for all possible (steering, speed) action combinations.
The architecture is designed for stable DQN learning with proper regularization.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_unified_model(input_dims, n_actions, learning_rate=0.001):
    """
    Build deep Q-network for unified racing agent.

    Architecture:
    - Input: Game state features (preprocessed image data)
    - Hidden layers: Deep network with batch normalization and dropout
    - Output: Q-values for each (steering, speed) action combination

    Parameters:
    - input_dims (int): Number of input features
    - n_actions (int): Number of discrete action combinations
    - learning_rate (float): Learning rate for optimizer

    Returns:
    - model (tf.keras.Model): Compiled Keras model
    """

    model = Sequential([
        # Input layer
        Dense(
            512,
            input_dim=input_dims,
            activation='relu',
            kernel_regularizer=l2(0.01),
            name='input_layer'
        ),
        BatchNormalization(),
        Dropout(0.2),

        # Hidden layer 1
        Dense(
            512,
            activation='relu',
            kernel_regularizer=l2(0.01),
            name='hidden_1'
        ),
        BatchNormalization(),
        Dropout(0.2),

        # Hidden layer 2
        Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(0.01),
            name='hidden_2'
        ),
        BatchNormalization(),
        Dropout(0.2),

        # Hidden layer 3
        Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(0.01),
            name='hidden_3'
        ),
        BatchNormalization(),

        # Output layer - Q-values for each action
        Dense(
            n_actions,
            activation='linear',
            name='q_values'
        )
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',  # Mean squared error for Q-learning
        metrics=['mae']  # Mean absolute error for monitoring
    )

    print(f"Unified Model Architecture:")
    print(f"  Input: {input_dims} features")
    print(f"  Output: {n_actions} Q-values (action combinations)")
    print(f"  Learning rate: {learning_rate}")
    model.summary()

    return model


def build_dueling_dqn_model(input_dims, n_actions, learning_rate=0.001):
    """
    Build Dueling DQN architecture for improved learning.

    Dueling DQN separates value and advantage streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    This architecture learns state values independently from action advantages,
    leading to better policy evaluation and faster learning.

    Parameters:
    - input_dims (int): Number of input features
    - n_actions (int): Number of discrete action combinations
    - learning_rate (float): Learning rate for optimizer

    Returns:
    - model (tf.keras.Model): Compiled Keras model with dueling architecture
    """

    # Input layer
    inputs = tf.keras.Input(shape=(input_dims,))

    # Shared feature extraction layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    # Value stream
    value_stream = Dense(256, activation='relu')(x)
    value_stream = Dense(1, activation='linear', name='state_value')(value_stream)

    # Advantage stream
    advantage_stream = Dense(256, activation='relu')(x)
    advantage_stream = Dense(n_actions, activation='linear', name='advantages')(advantage_stream)

    # Combine value and advantage to get Q-values
    # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
    mean_advantage = tf.reduce_mean(advantage_stream, axis=1, keepdims=True)
    q_values = value_stream + (advantage_stream - mean_advantage)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=q_values, name='dueling_dqn')

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    print(f"Dueling DQN Model Architecture:")
    print(f"  Input: {input_dims} features")
    print(f"  Output: {n_actions} Q-values (dueling streams)")
    print(f"  Learning rate: {learning_rate}")
    model.summary()

    return model
