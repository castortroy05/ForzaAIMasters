"""
Unified Racing Agent - Main Training Script

This is the new main entry point for training the unified racing agent.
It replaces the old dual-agent system with a single coordinated agent.

Usage:
    python main_unified.py

Features:
- Single unified agent controlling both steering and speed
- Progressive learning from novice to pro level
- Curriculum learning support
- Automatic checkpointing and recovery
- Performance monitoring and visualization
"""

import sys
import os
import time

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity
import tensorflow as tf

# Add the 'game' directory to the system path
sys.path.append(os.path.join(os.getcwd(), 'src/game/'))

# Import unified agent components
from game_env import GameEnvironment
from unified_agent import UnifiedRacingAgent
from unified_training import train_racing_agent, evaluate_agent

# Window management
import pygetwindow as gw


def is_game_running(window_title="Forza Motorsport 7"):
    """
    Check if the game window is currently running.

    Parameters:
    - window_title (str): The title of the game window

    Returns:
    - bool: True if game window is found
    """
    try:
        windows = gw.getWindowsWithTitle(window_title)
        return len(windows) > 0
    except Exception as e:
        print(f"Warning: Could not check game window: {e}")
        return False


def print_header():
    """Print welcome header."""
    print("\n" + "="*80)
    print("🏎️  FORZA MOTORSPORT 7 - UNIFIED RACING AGENT  🏁".center(80))
    print("Deep Q-Learning with Coordinated Control".center(80))
    print("="*80 + "\n")


def print_menu():
    """Print main menu options with improved visual organization."""
    print("\n" + "─"*80)
    print("  SELECT OPTION".center(80))
    print("─"*80 + "\n")

    # Training Options
    print("┌─ TRAINING ─────────────────────────────────────────────────────────┐")
    print("│                                                                      │")
    print("│  1. 🎓 Train New Agent             Novice → Pro (default 1000 eps)  │")
    print("│  2. 🔄 Continue from Checkpoint    Resume previous training         │")
    print("│  4. ⚡ Quick Training              Fast mode (100 episodes)         │")
    print("│  5. 🏆 Full Training               Complete training (1000 eps)     │")
    print("│                                                                      │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    print()

    # Evaluation & Management
    print("┌─ EVALUATION & MANAGEMENT ──────────────────────────────────────────┐")
    print("│                                                                      │")
    print("│  3. 📈 Evaluate Trained Agent      Test your best model             │")
    print("│  6. ℹ️  System Information          Check GPU/TensorFlow            │")
    print("│  7. 🚪 Exit                         Quit application                │")
    print("│                                                                      │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    print()

    print("💡 TIP: Start with option 1 to train a new agent from scratch")
    print()


def get_system_info():
    """Display system information."""
    print("\n" + "-"*80)
    print("SYSTEM INFORMATION".center(80))
    print("-"*80)

    # GPU information
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"\nGPUs Available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # CPU information
    print(f"\nCPUs Available: {len(tf.config.experimental.list_physical_devices('CPU'))}")

    # TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Game window check
    game_running = is_game_running()
    print(f"\nGame Status: {'RUNNING ✓' if game_running else 'NOT DETECTED ✗'}")

    print("-"*80 + "\n")


def create_agent(game_env, input_dims=6):
    """
    Create a new unified racing agent with optimized hyperparameters.

    Parameters:
    - game_env: Game environment instance
    - input_dims (int): Number of input features

    Returns:
    - UnifiedRacingAgent: Configured agent instance
    """
    print("\nInitializing Unified Racing Agent...")

    agent = UnifiedRacingAgent(
        game_env=game_env,
        input_dims=input_dims,
        mem_size=50000,          # Large replay buffer
        batch_size=64,           # Standard batch size
        eps=1.0,                 # Start with full exploration
        eps_min=0.01,            # Minimum 1% exploration
        eps_dec=0.995,           # Slow epsilon decay (per episode)
        gamma=0.99,              # Standard discount factor
        learning_rate=0.001,     # Conservative learning rate
        target_update_freq=1000, # Update target network every 1000 steps
        model_save_dir='models/unified_agent'
    )

    print("Agent initialized successfully!\n")
    return agent


def train_new_agent(game_env, num_episodes=1000):
    """
    Train a new agent from scratch.

    Parameters:
    - game_env: Game environment instance
    - num_episodes (int): Number of training episodes
    """
    print("\n" + "="*80)
    print("TRAINING NEW AGENT".center(80))
    print("="*80 + "\n")

    # Create agent
    agent = create_agent(game_env)

    # Train agent
    print(f"Starting training for {num_episodes} episodes...")
    print("The agent will learn progressively from novice to pro level.")
    print("Checkpoints will be saved every 50 episodes.\n")

    input("Press ENTER when ready to start training...")

    history = train_racing_agent(
        agent=agent,
        num_episodes=num_episodes,
        save_frequency=50,
        verbose=True,
        use_curriculum=True,
        visualization=True
    )

    print("\nTraining complete!")
    print(f"Final best reward: {history['best_reward']:.2f}")

    return agent, history


def continue_training(game_env):
    """
    Continue training from a saved checkpoint.

    Parameters:
    - game_env: Game environment instance
    """
    print("\n" + "="*80)
    print("CONTINUE TRAINING".center(80))
    print("="*80 + "\n")

    # List available checkpoints
    model_dir = 'models/unified_agent'
    if not os.path.exists(model_dir):
        print("No checkpoints found. Please train a new agent first.")
        return

    # Find checkpoint files
    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_episode_') and f.endswith('.h5')]

    if not checkpoint_files:
        print("No checkpoint files found.")
        return

    # Sort by episode number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

    print("Available checkpoints:")
    for i, f in enumerate(checkpoint_files):
        episode = f.split('_')[2].split('.')[0]
        print(f"  {i+1}. Episode {episode}")

    choice = input("\nSelect checkpoint number (or press ENTER for latest): ")

    if choice.strip() == "":
        checkpoint_file = checkpoint_files[-1]
    else:
        try:
            idx = int(choice) - 1
            checkpoint_file = checkpoint_files[idx]
        except:
            print("Invalid choice.")
            return

    # Load agent
    print(f"\nLoading checkpoint: {checkpoint_file}")
    agent = create_agent(game_env)

    try:
        checkpoint_path = os.path.join(model_dir, checkpoint_file)
        agent.load_model(checkpoint_path)

        # Try to load replay buffer
        try:
            episode_num = checkpoint_file.split('_')[2].split('.')[0]
            replay_file = f'replay_buffer_episode_{episode_num}.pkl'
            agent.load_memory(replay_file)
        except (IndexError, ValueError) as e:
            print(f"Could not parse episode number from filename: {e}")
            print("Could not load replay buffer (starting fresh)")

        print("Checkpoint loaded successfully!")

        # Continue training
        num_episodes = int(input("\nHow many additional episodes to train? "))

        history = train_racing_agent(
            agent=agent,
            num_episodes=num_episodes,
            save_frequency=50,
            verbose=True,
            use_curriculum=True,
            visualization=True
        )

        print("\nContinued training complete!")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def evaluate_trained_agent(game_env):
    """
    Evaluate a trained agent.

    Parameters:
    - game_env: Game environment instance
    """
    print("\n" + "="*80)
    print("AGENT EVALUATION".center(80))
    print("="*80 + "\n")

    # Find best model
    model_dir = 'models/unified_agent'
    best_model_path = os.path.join(model_dir, 'best_model.h5')

    if not os.path.exists(best_model_path):
        print("No trained model found. Please train an agent first.")
        return

    # Load agent
    print("Loading best model...")
    agent = create_agent(game_env)

    try:
        agent.load_model(best_model_path)
        print("Model loaded successfully!\n")

        num_episodes = int(input("Number of evaluation episodes (default 10): ") or "10")

        # Evaluate
        results = evaluate_agent(agent, num_episodes=num_episodes, render=True)

        print("\nEvaluation complete!")

    except Exception as e:
        print(f"Error loading model: {e}")


def main():
    """Main function."""
    print_header()

    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUs Available: {len(gpus)}")
    if gpus:
        print("GPU acceleration enabled ✓")
    else:
        print("Running on CPU (training will be slower)")

    print()

    while True:
        print_menu()
        choice = input("Enter your choice (1-7): ")

        # Check if game is running for training/evaluation
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

            # Initialize game environment
            try:
                print("\nInitializing game environment...")
                game_env = GameEnvironment()
                print("Game environment initialized ✓\n")
            except Exception as e:
                print(f"\nError initializing game environment: {e}")
                print("Please check that the game window is accessible.")
                continue

        # Handle menu choices
        if choice == "1":
            # Train new agent
            num_episodes = int(input("\nNumber of episodes (default 1000): ") or "1000")
            train_new_agent(game_env, num_episodes)

        elif choice == "2":
            # Continue training
            continue_training(game_env)

        elif choice == "3":
            # Evaluate agent
            evaluate_trained_agent(game_env)

        elif choice == "4":
            # Quick training
            print("\nQuick Training Mode (100 episodes)")
            train_new_agent(game_env, num_episodes=100)

        elif choice == "5":
            # Full training
            print("\nFull Training Mode (1000 episodes)")
            print("This will take several hours depending on your hardware.")
            confirm = input("Continue? (y/n): ")
            if confirm.lower() == 'y':
                train_new_agent(game_env, num_episodes=1000)

        elif choice == "6":
            # System info
            get_system_info()

        elif choice == "7":
            # Exit
            print("\n" + "="*80)
            print("Thank you for using the Unified Racing Agent!".center(80))
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
