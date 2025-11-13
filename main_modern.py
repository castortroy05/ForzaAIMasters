"""
Modern Racing Agent - 2024-2025 State-of-the-Art

This is the cutting-edge version using:
- Vision Transformers / EfficientNet for visual processing
- PPO (Proximal Policy Optimization) for superior continuous control
- Intrinsic curiosity for exploration
- Modern training infrastructure (TensorBoard, LR scheduling, etc.)
- Attention mechanisms for track awareness
- Temporal reasoning with LSTM/Transformer

Usage:
    python main_modern.py
"""

import sys
import os
import numpy as np

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Add game directory to path
sys.path.append(os.path.join(os.getcwd(), 'src/game/'))

# Modern imports
from game_env import GameEnvironment
from modern_vision import ModernVisionEncoder, create_modern_state_encoder
from modern_ppo_agent import ModernPPOAgent
from modern_training import ModernTrainer

import pygetwindow as gw


def print_header():
    """Print welcome header."""
    print("\n" + "="*80)
    print("FORZA MOTORSPORT 7 - MODERN RACING AGENT (2024-2025)".center(80))
    print("State-of-the-Art Deep Reinforcement Learning".center(80))
    print("="*80 + "\n")


def print_menu():
    """Print main menu."""
    print("\nChoose training mode:")
    print("1. Train with Vision Transformer (ViT) - Attention-based")
    print("2. Train with EfficientNetV2 - Fast & efficient")
    print("3. Train with ConvNeXt - Modern CNN")
    print("4. Train with Simple Features (fast, for testing)")
    print("5. Evaluate Trained Model")
    print("6. System Information")
    print("7. Exit")
    print()


def get_system_info():
    """Display system information."""
    print("\n" + "-"*80)
    print("SYSTEM INFORMATION".center(80))
    print("-"*80)

    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs Available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        # Get GPU memory
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Compute Capability: {details.get('compute_capability', 'Unknown')}")
        except:
            pass

    # Mixed precision
    print(f"\nMixed Precision: {'Available' if gpus else 'Not available (CPU only)'}")

    # TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Check TensorFlow Probability
    try:
        import tensorflow_probability as tfp
        print(f"TensorFlow Probability: {tfp.__version__} ✓")
    except ImportError:
        print("TensorFlow Probability: Not installed ✗")
        print("  Install with: pip install tensorflow-probability")

    # Game window
    try:
        windows = gw.getWindowsWithTitle("Forza Motorsport 7")
        print(f"\nGame Status: {'RUNNING ✓' if windows else 'NOT DETECTED ✗'}")
    except:
        print("\nGame Status: Could not check")

    print("-"*80 + "\n")


def is_game_running(window_title="Forza Motorsport 7"):
    """Check if game window is running."""
    try:
        windows = gw.getWindowsWithTitle(window_title)
        return len(windows) > 0
    except Exception as e:
        print(f"Warning: Could not check game window: {e}")
        return False


def train_modern_agent(architecture='efficientnet', num_episodes=1000):
    """
    Train modern PPO agent with visual encoder.

    Parameters:
    - architecture: 'vit', 'efficientnet', or 'convnext'
    - num_episodes: Number of training episodes
    """
    print("\n" + "="*80)
    print(f"TRAINING MODERN AGENT - {architecture.upper()}".center(80))
    print("="*80 + "\n")

    # Initialize game environment
    print("Initializing game environment...")
    game_env = GameEnvironment()
    print("Game environment initialized ✓\n")

    # Create vision encoder
    print(f"Building {architecture.upper()} vision encoder...")
    vision_encoder = create_modern_state_encoder(
        input_shape=(240, 320, 3),  # Adjust to your game resolution
        architecture=architecture
    )
    print("Vision encoder ready ✓\n")

    # Create PPO agent
    print("Initializing PPO agent...")
    agent = ModernPPOAgent(
        state_dim=512,  # Vision encoder output dimension
        action_dim=2,   # [steering, throttle/brake]
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        buffer_size=2048,
        mini_batch_size=64,
        epochs_per_update=10,
        use_mixed_precision=True
    )
    print("PPO agent ready ✓\n")

    # Create modern trainer
    print("Setting up modern training infrastructure...")
    trainer = ModernTrainer(
        agent=agent,
        game_env=game_env,
        vision_encoder=vision_encoder,
        use_curiosity=True,
        curiosity_weight=0.1,
        use_tensorboard=True
    )
    print("Trainer ready ✓\n")

    print(f"Starting training for {num_episodes} episodes...")
    print("=" * 80)

    input("Press ENTER to start training...")

    # Train
    history = trainer.train(num_episodes=num_episodes, eval_frequency=100)

    print("\nTraining complete!")
    print(f"Best reward achieved: {history['best_reward']:.2f}")

    return agent, history


def train_simple_agent(num_episodes=1000):
    """
    Train with simple feature extraction (for testing/debugging).
    """
    print("\n" + "="*80)
    print("TRAINING WITH SIMPLE FEATURES (Fast Mode)".center(80))
    print("="*80 + "\n")

    # Initialize game environment
    game_env = GameEnvironment()

    # Create PPO agent with simple features
    agent = ModernPPOAgent(
        state_dim=6,  # Simple mean/std features
        action_dim=2,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=2048,
        use_mixed_precision=False  # Disable for CPU
    )

    # Create trainer (no vision encoder)
    trainer = ModernTrainer(
        agent=agent,
        game_env=game_env,
        vision_encoder=None,
        use_curiosity=False,
        use_tensorboard=True
    )

    input("Press ENTER to start training...")

    # Train
    history = trainer.train(num_episodes=num_episodes)

    print(f"\nBest reward: {history['best_reward']:.2f}")

    return agent, history


def evaluate_model():
    """Evaluate a trained model."""
    print("\n" + "="*80)
    print("MODEL EVALUATION".center(80))
    print("="*80 + "\n")

    model_path = input("Enter model path (or press ENTER for best_model): ").strip()
    if not model_path:
        model_path = 'best_model'

    architecture = input("Vision architecture used (vit/efficientnet/convnext/simple): ").strip().lower()

    # Initialize environment
    game_env = GameEnvironment()

    # Create vision encoder if needed
    if architecture != 'simple':
        vision_encoder = create_modern_state_encoder(
            input_shape=(240, 320, 3),
            architecture=architecture
        )
        state_dim = 512
    else:
        vision_encoder = None
        state_dim = 6

    # Create agent
    agent = ModernPPOAgent(state_dim=state_dim, action_dim=2)

    # Load model
    try:
        agent.load(filename=model_path)
        print(f"Model loaded: {model_path} ✓\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate
    num_eval = int(input("Number of evaluation episodes (default 10): ") or "10")

    trainer = ModernTrainer(
        agent=agent,
        game_env=game_env,
        vision_encoder=vision_encoder,
        use_curiosity=False
    )

    trainer._evaluate(num_eval_episodes=num_eval)


def main():
    """Main function."""
    print_header()

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) detected ✓")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Running on CPU (training will be slower)")

    print()

    while True:
        print_menu()
        choice = input("Enter your choice (1-7): ").strip()

        # Check game running for training/eval
        if choice in ["1", "2", "3", "4", "5"]:
            if not is_game_running("Forza Motorsport 7"):
                print("\n⚠ WARNING: Game 'Forza Motorsport 7' not detected!")
                print("Please ensure:")
                print("  1. The game is running")
                print("  2. The game is in windowed mode")
                print("  3. The game window title contains 'Forza Motorsport 7'")

                proceed = input("\nProceed anyway? (y/n): ")
                if proceed.lower() != 'y':
                    continue

        if choice == "1":
            # Vision Transformer
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_modern_agent(architecture='vit', num_episodes=num_episodes)

        elif choice == "2":
            # EfficientNet
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_modern_agent(architecture='efficientnet', num_episodes=num_episodes)

        elif choice == "3":
            # ConvNeXt
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_modern_agent(architecture='convnext', num_episodes=num_episodes)

        elif choice == "4":
            # Simple features
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_simple_agent(num_episodes=num_episodes)

        elif choice == "5":
            # Evaluate
            evaluate_model()

        elif choice == "6":
            # System info
            get_system_info()

        elif choice == "7":
            # Exit
            print("\n" + "="*80)
            print("Thank you for using Modern Racing Agent!".center(80))
            print("="*80 + "\n")
            break

        else:
            print("\n⚠ Invalid choice. Please select 1-7.")

        # Pause before returning to menu
        if choice in ["1", "2", "3", "4", "5"]:
            input("\nPress ENTER to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        print("Goodbye!")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
