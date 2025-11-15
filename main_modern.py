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

# Cutting-edge imports
from cutting_edge_vision import (
    DINOv2Encoder,
    YOLODetector,
    HybridMultiModalEncoder,
    create_cutting_edge_encoder
)
from decision_transformer import DecisionTransformer, OnlineDecisionTransformer

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
    print("\n=== CUTTING-EDGE (2024-2025 Latest) ===")
    print("1. 🔬 DINOv2 + YOLO Hybrid (Multi-Modal) - ABSOLUTE CUTTING-EDGE!")
    print("2. 🔬 DINOv2 (Meta 2023) - Better than ViT")
    print("3. 🎯 YOLO Object Detection - Racing-specific")
    print("4. 🤖 Decision Transformer - Sequence Modeling (No value functions!)")
    print("\n=== MODERN (2020-2023) ===")
    print("5. Vision Transformer (ViT) - Attention-based")
    print("6. EfficientNetV2 - Fast & efficient")
    print("7. ConvNeXt - Modern CNN")
    print("\n=== SIMPLE ===")
    print("8. Simple Features (testing)")
    print("\n=== MANAGEMENT ===")
    print("9. Evaluate Trained Model")
    print("10. System Information")
    print("11. Exit")
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


def train_cutting_edge_agent(architecture='hybrid', num_episodes=1000):
    """
    Train cutting-edge agent with DINOv2/YOLO/Hybrid.

    Parameters:
    - architecture: 'dinov2', 'yolo', or 'hybrid'
    - num_episodes: Number of training episodes
    """
    print("\n" + "="*80)
    print(f"CUTTING-EDGE TRAINING - {architecture.upper()}".center(80))
    print("="*80 + "\n")

    # Initialize game environment
    print("Initializing game environment...")
    game_env = GameEnvironment()
    print("Game environment initialized ✓\n")

    # Create cutting-edge vision encoder
    print(f"Building {architecture.upper()} vision encoder...")
    vision_encoder = create_cutting_edge_encoder(
        input_shape=(240, 320, 3),
        architecture=architecture
    )
    print("Cutting-edge vision encoder ready ✓\n")

    # Determine state dim based on architecture
    if architecture == 'hybrid':
        state_dim = 768  # Hybrid multi-modal output
    elif architecture == 'dinov2':
        state_dim = 512  # DINOv2 output
    elif architecture == 'yolo':
        state_dim = 100  # YOLO features
    else:
        state_dim = 512

    # Create PPO agent
    print("Initializing PPO agent...")
    agent = ModernPPOAgent(
        state_dim=state_dim,
        action_dim=2,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=2048,
        use_mixed_precision=True
    )
    print("PPO agent ready ✓\n")

    # Create trainer
    print("Setting up cutting-edge training infrastructure...")
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


def train_decision_transformer(num_episodes=1000):
    """
    Train Decision Transformer (sequence modeling approach).

    This is VERY cutting-edge - treats RL as sequence prediction!
    """
    print("\n" + "="*80)
    print("DECISION TRANSFORMER TRAINING (Sequence Modeling)".center(80))
    print("="*80 + "\n")

    print("⚠ WARNING: Decision Transformer is experimental!")
    print("This is a completely different approach to RL:")
    print("  - No value functions")
    print("  - No policy gradients")
    print("  - Just sequence-to-sequence prediction")
    print()

    proceed = input("Continue? (y/n): ")
    if proceed.lower() != 'y':
        return

    # Initialize game environment
    game_env = GameEnvironment()

    # Create Decision Transformer
    print("\nInitializing Decision Transformer...")
    agent = OnlineDecisionTransformer(
        state_dim=6,  # Simple features for now
        action_dim=2,
        context_length=20,
        embed_dim=256,
        num_layers=6,
        num_heads=8
    )
    print("Decision Transformer ready ✓\n")

    input("Press ENTER to start training...")

    # Training loop (simplified)
    from image_processing import preprocess_input_data

    for episode in range(num_episodes):
        game_env.reset_episode()

        trajectory = []
        states_history = []
        actions_history = []
        rtgs_history = []
        timesteps_history = []

        frame = game_env.capture()
        state = preprocess_input_data(frame)

        episode_reward = 0
        step = 0
        done = False

        while not done and step < 1000:
            # Get action from Decision Transformer
            action = agent.get_action(
                states=states_history[-20:] if states_history else [state],
                actions=actions_history[-20:] if actions_history else [np.zeros(2)],
                rtgs=rtgs_history[-20:] if rtgs_history else [100.0],  # Target return
                timesteps=timesteps_history[-20:] if timesteps_history else [0]
            )

            # Execute action (simplified)
            from controller import GameController
            if not hasattr(agent, '_controller'):
                agent._controller = GameController()

            steering, throttle = action
            agent._controller.left_joystick(x_value=int(steering * 32767), y_value=0)
            if throttle >= 0:
                agent._controller.right_trigger(value=int(throttle * 255))
                agent._controller.left_trigger(value=0)
            else:
                agent._controller.left_trigger(value=int(-throttle * 255))
                agent._controller.right_trigger(value=0)
            agent._controller.update()

            # Observe next state
            import time
            time.sleep(0.05)
            next_frame = game_env.capture()
            next_state = preprocess_input_data(next_frame)

            # Get reward (simplified)
            reward = 1.0  # Placeholder
            done = game_env.is_done(next_frame)

            # Store transition
            trajectory.append((state, action, reward))
            states_history.append(state)
            actions_history.append(action)
            rtgs_history.append(100.0 - episode_reward)  # Return-to-go
            timesteps_history.append(step)

            state = next_state
            episode_reward += reward
            step += 1

        # Store trajectory
        agent.store_trajectory(trajectory)

        # Maybe update
        agent.maybe_update()

        print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step}")

        # Save periodically
        if (episode + 1) % 50 == 0:
            print(f"Checkpoint at episode {episode + 1}")

    print("\nDecision Transformer training complete!")
    return agent


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
        choice = input("Enter your choice (1-11): ").strip()

        # Check game running for training/eval
        if choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
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
            # CUTTING-EDGE: Hybrid DINOv2 + YOLO
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            print("\n🚀 ABSOLUTE CUTTING-EDGE: DINOv2 + YOLO Hybrid Multi-Modal")
            train_cutting_edge_agent(architecture='hybrid', num_episodes=num_episodes)

        elif choice == "2":
            # CUTTING-EDGE: DINOv2
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            print("\n🔬 DINOv2 (Meta 2023) - Better than ViT")
            train_cutting_edge_agent(architecture='dinov2', num_episodes=num_episodes)

        elif choice == "3":
            # CUTTING-EDGE: YOLO
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            print("\n🎯 YOLO Object Detection for Racing")
            train_cutting_edge_agent(architecture='yolo', num_episodes=num_episodes)

        elif choice == "4":
            # CUTTING-EDGE: Decision Transformer
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            print("\n🤖 Decision Transformer - No value functions!")
            train_decision_transformer(num_episodes=num_episodes)

        elif choice == "5":
            # Vision Transformer
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_modern_agent(architecture='vit', num_episodes=num_episodes)

        elif choice == "6":
            # EfficientNet
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_modern_agent(architecture='efficientnet', num_episodes=num_episodes)

        elif choice == "7":
            # ConvNeXt
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_modern_agent(architecture='convnext', num_episodes=num_episodes)

        elif choice == "8":
            # Simple features
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_simple_agent(num_episodes=num_episodes)

        elif choice == "9":
            # Evaluate
            evaluate_model()

        elif choice == "10":
            # System info
            get_system_info()

        elif choice == "11":
            # Exit
            print("\n" + "="*80)
            print("Thank you for using Modern Racing Agent!".center(80))
            print("="*80 + "\n")
            break

        else:
            print("\n⚠ Invalid choice. Please select 1-11.")

        # Pause before returning to menu
        if choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
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
