"""
Modern Vision Encoder for Racing Agent (2024-2025 Techniques)

Uses state-of-the-art computer vision models:
- EfficientNetV2 for efficient visual feature extraction
- Vision Transformer (ViT) option for attention-based processing
- Spatial attention mechanisms for track/chevron focus
- Multi-scale feature extraction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ModernVisionEncoder:
    """
    Modern vision encoder using EfficientNetV2 or Vision Transformer.

    Replaces simple mean/std feature extraction with deep CNN features.
    """

    def __init__(self, input_shape=(240, 320, 3), architecture='efficientnet',
                 feature_dim=512, use_attention=True):
        """
        Initialize modern vision encoder.

        Parameters:
        - input_shape: Input image shape (H, W, C)
        - architecture: 'efficientnet', 'vit', or 'convnext'
        - feature_dim: Output feature dimension
        - use_attention: Whether to use spatial attention
        """
        self.input_shape = input_shape
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.use_attention = use_attention

        self.encoder = self._build_encoder()

    def _build_encoder(self):
        """Build the vision encoder network."""

        inputs = layers.Input(shape=self.input_shape, name='image_input')

        # Preprocessing
        x = layers.Rescaling(1./255)(inputs)

        # Choose backbone
        if self.architecture == 'efficientnet':
            x = self._efficientnet_backbone(x)
        elif self.architecture == 'vit':
            x = self._vit_backbone(x)
        elif self.architecture == 'convnext':
            x = self._convnext_backbone(x)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        # Optional spatial attention
        if self.use_attention:
            x = self._spatial_attention(x)

        # Global pooling and feature projection
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(self.feature_dim, activation='gelu', name='feature_projection')(x)
        x = layers.LayerNormalization()(x)

        outputs = x

        model = keras.Model(inputs=inputs, outputs=outputs, name=f'{self.architecture}_encoder')
        return model

    def _efficientnet_backbone(self, x):
        """EfficientNetV2 backbone - fast and accurate."""
        # Load pretrained EfficientNetV2B0 (can be trained from scratch too)
        base_model = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,  # Train from scratch for game-specific features
            input_tensor=x,
            pooling=None
        )
        return base_model.output

    def _convnext_backbone(self, x):
        """ConvNeXt backbone - modern CNN architecture."""
        # ConvNeXt block implementation
        def convnext_block(x, filters, kernel_size=7):
            residual = x

            # Depthwise convolution
            x = layers.DepthwiseConv2D(kernel_size, padding='same')(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            # Pointwise expansion
            x = layers.Dense(filters * 4, activation='gelu')(x)
            x = layers.Dense(filters)(x)

            # Residual connection
            x = layers.Add()([residual, x])
            return x

        # Initial convolution
        x = layers.Conv2D(96, 4, strides=4, padding='same')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # ConvNeXt blocks
        for _ in range(3):
            x = convnext_block(x, 96)

        # Downsample
        x = layers.Conv2D(192, 2, strides=2, padding='same')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        for _ in range(3):
            x = convnext_block(x, 192)

        return x

    def _vit_backbone(self, x):
        """Vision Transformer backbone - attention-based."""
        patch_size = 16
        num_patches = (self.input_shape[0] // patch_size) * (self.input_shape[1] // patch_size)
        projection_dim = 256
        num_heads = 8
        num_layers = 6

        # Patch extraction
        patches = layers.Conv2D(projection_dim, patch_size, strides=patch_size, padding='valid')(x)
        patch_dims = patches.shape[1:3]
        patches = layers.Reshape((patch_dims[0] * patch_dims[1], projection_dim))(patches)

        # Position embeddings
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
        patches = patches + pos_embedding

        # Transformer blocks
        for _ in range(num_layers):
            # Multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(patches, patches)

            # Skip connection and layer norm
            x1 = layers.Add()([patches, attn_output])
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)

            # MLP
            x2 = layers.Dense(projection_dim * 4, activation='gelu')(x1)
            x2 = layers.Dropout(0.1)(x2)
            x2 = layers.Dense(projection_dim)(x2)

            # Skip connection and layer norm
            patches = layers.Add()([x1, x2])
            patches = layers.LayerNormalization(epsilon=1e-6)(patches)

        # Reshape back to spatial for compatibility
        spatial_dim = int(np.sqrt(num_patches))
        x = layers.Reshape((spatial_dim, spatial_dim, projection_dim))(patches)

        return x

    def _spatial_attention(self, x):
        """
        Spatial attention mechanism to focus on important regions (track, chevrons).
        """
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
        max_pool = layers.GlobalMaxPooling2D(keepdims=True)(x)

        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(concat)

        x = layers.Multiply()([x, attention])

        # Spatial attention
        channel_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        channel_max = tf.reduce_max(x, axis=-1, keepdims=True)

        spatial_concat = layers.Concatenate(axis=-1)([channel_avg, channel_max])
        spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)

        x = layers.Multiply()([x, spatial_attention])

        return x

    def encode(self, image):
        """
        Encode image to feature vector.

        Parameters:
        - image: numpy array (H, W, C)

        Returns:
        - features: numpy array (feature_dim,)
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        features = self.encoder.predict(image, verbose=0)[0]
        return features

    def get_model(self):
        """Get the encoder model."""
        return self.encoder


def create_modern_state_encoder(input_shape=(240, 320, 3), architecture='efficientnet'):
    """
    Factory function to create modern state encoder.

    Parameters:
    - input_shape: Image input shape
    - architecture: 'efficientnet', 'vit', or 'convnext'

    Returns:
    - encoder: ModernVisionEncoder instance
    """
    encoder = ModernVisionEncoder(
        input_shape=input_shape,
        architecture=architecture,
        feature_dim=512,
        use_attention=True
    )

    print(f"\nModern Vision Encoder Created:")
    print(f"  Architecture: {architecture.upper()}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output features: 512")
    print(f"  Spatial attention: Enabled")

    return encoder


class TemporalFeatureExtractor:
    """
    Adds temporal reasoning using LSTM or Transformer.

    Processes sequences of visual features to understand motion, speed, trajectory.
    """

    def __init__(self, feature_dim=512, temporal_dim=256, sequence_length=4,
                 architecture='lstm'):
        """
        Initialize temporal feature extractor.

        Parameters:
        - feature_dim: Input feature dimension
        - temporal_dim: Output temporal feature dimension
        - sequence_length: Number of frames to process
        - architecture: 'lstm', 'gru', or 'transformer'
        """
        self.feature_dim = feature_dim
        self.temporal_dim = temporal_dim
        self.sequence_length = sequence_length
        self.architecture = architecture

        self.model = self._build_model()

    def _build_model(self):
        """Build temporal processing model."""
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim),
                             name='temporal_input')

        if self.architecture == 'lstm':
            x = layers.LSTM(self.temporal_dim, return_sequences=False)(inputs)
        elif self.architecture == 'gru':
            x = layers.GRU(self.temporal_dim, return_sequences=False)(inputs)
        elif self.architecture == 'transformer':
            # Transformer encoder
            x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
            x = layers.Add()([inputs, x])
            x = layers.LayerNormalization()(x)

            # Feedforward
            ff = layers.Dense(self.temporal_dim * 4, activation='gelu')(x)
            ff = layers.Dense(self.temporal_dim)(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)

            # Global pooling
            x = layers.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        x = layers.Dense(self.temporal_dim, activation='gelu')(x)
        x = layers.LayerNormalization()(x)

        model = keras.Model(inputs=inputs, outputs=x,
                           name=f'{self.architecture}_temporal')
        return model

    def extract_temporal_features(self, feature_sequence):
        """
        Extract temporal features from sequence.

        Parameters:
        - feature_sequence: numpy array (sequence_length, feature_dim)

        Returns:
        - temporal_features: numpy array (temporal_dim,)
        """
        if len(feature_sequence.shape) == 2:
            feature_sequence = np.expand_dims(feature_sequence, axis=0)

        features = self.model.predict(feature_sequence, verbose=0)[0]
        return features


class CuriosityModule:
    """
    Intrinsic motivation module for exploration.

    Implements ICM (Intrinsic Curiosity Module) to encourage exploration
    of novel states and behaviors.
    """

    def __init__(self, state_dim, action_dim, feature_dim=256):
        """
        Initialize curiosity module.

        Parameters:
        - state_dim: State space dimension
        - action_dim: Action space dimension
        - feature_dim: Feature encoding dimension
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.feature_encoder = self._build_feature_encoder()
        self.forward_model = self._build_forward_model()
        self.inverse_model = self._build_inverse_model()

    def _build_feature_encoder(self):
        """Encode states to feature space."""
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='gelu')(inputs)
        x = layers.Dense(self.feature_dim, activation='gelu')(x)
        x = layers.LayerNormalization()(x)

        return keras.Model(inputs=inputs, outputs=x, name='feature_encoder')

    def _build_forward_model(self):
        """Predict next state features given current features and action."""
        state_input = layers.Input(shape=(self.feature_dim,))
        action_input = layers.Input(shape=(self.action_dim,))

        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dense(self.feature_dim)(x)

        return keras.Model(inputs=[state_input, action_input], outputs=x,
                          name='forward_model')

    def _build_inverse_model(self):
        """Predict action given current and next state features."""
        current_state = layers.Input(shape=(self.feature_dim,))
        next_state = layers.Input(shape=(self.feature_dim,))

        x = layers.Concatenate()([current_state, next_state])
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dense(self.action_dim, activation='softmax')(x)

        return keras.Model(inputs=[current_state, next_state], outputs=x,
                          name='inverse_model')

    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic reward based on prediction error.

        Higher reward for surprising/novel transitions.

        Parameters:
        - state: Current state
        - action: Action taken
        - next_state: Next state

        Returns:
        - intrinsic_reward: Curiosity-driven reward
        """
        # Encode states
        state_features = self.feature_encoder.predict(np.array([state]), verbose=0)[0]
        next_state_features = self.feature_encoder.predict(np.array([next_state]), verbose=0)[0]

        # Predict next state features
        action_onehot = np.zeros(self.action_dim)
        if isinstance(action, int):
            action_onehot[action] = 1
        else:
            action_onehot = action

        predicted_next = self.forward_model.predict(
            [np.array([state_features]), np.array([action_onehot])],
            verbose=0
        )[0]

        # Intrinsic reward is prediction error
        intrinsic_reward = np.mean(np.square(predicted_next - next_state_features))

        return intrinsic_reward
