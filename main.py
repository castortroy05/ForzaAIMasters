import sys
import os
#show full Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU to use (0, 1) for multiple GPUs

sys.path.append(os.path.join(os.getcwd(), 'src/game/'))  # Add the 'game' folder to the system path

import agent_steering  # Import the steering agent
import agent_speed  # Import the speed agent

import pygetwindow as gw
import nbformat
from IPython.core.interactiveshell import InteractiveShell

def is_game_running(window_title):
    """Check if the game window is currently running."""
    try:
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            return True
        else:
            return False
    except:
        return False

def run_notebook(notebook_path):
    """Run a Jupyter notebook."""
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    shell = InteractiveShell.instance()
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            shell.run_cell(cell.source)

def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
        print("Welcome to the Forza Motorsport 7 RL Trainer!")
        print("This program will train a reinforcement learning agent to control the vehicle in the game.")
        print("Before you start, please make sure that the game is running in the foreground.")
        print("Also, please make sure that the game is running in windowed mode.")
        print("If you are ready, please choose an option:")
        print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Please choose an option:")
        print("1. Train Steering Agent")
        print("2. Train Speed Agent")
        print("3. Train Both Agents")
        print("4. Test Controller")
        print("5. Exit")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice in ["1", "2", "3"]:
            if not is_game_running("Forza Motorsport 7"):  # Replace with the exact window title of the game
                print("The game is not currently running. Please start the game before training.")
                continue

            if choice == "1":
                # Initialize and train the steering agent
                steering_agent = agent_steering.Agent(
                    input_dims=6,
                    n_actions=2, mem_size=1000,
                    eps=1.0,
                    eps_min=0.01,
                    eps_dec=0.995,     
                    gamma=0.99,     
                    q_eval_name="q_eval_model",
                    q_target_name="q_target_model",
                    replace=100
                    )
                steering_agent.train(num_episodes=100)  # Specify number of episodes or other parameters

            elif choice == "2":
                # Initialize and train the speed agent
                speed_agent = agent_speed.Agent(
                    input_dims=6,
                    n_actions=2, mem_size=1000,
                    eps=1.0,
                    eps_min=0.01,
                    eps_dec=0.995,     
                    gamma=0.99,     
                    q_eval_name="q_eval_model",
                    q_target_name="q_target_model",
                    replace=100
                    )
                speed_agent.train(num_episodes=100)  # Specify number of episodes or other parameters

            elif choice == "3":
                # Initialize and train both agents
                steering_agent = agent_steering.Agent(
                    input_dims=6,
                    n_actions=2, mem_size=1000,
                    eps=1.0,
                    eps_min=0.01,
                    eps_dec=0.995,     
                    gamma=0.99,     
                    q_eval_name="q_eval_model",
                    q_target_name="q_target_model",
                    replace=100
                    )
                speed_agent = agent_speed.Agent(
                    input_dims=6,
                    n_actions=2, mem_size=1000,
                    eps=1.0,
                    eps_min=0.01,
                    eps_dec=0.995,     
                    gamma=0.99,     
                    q_eval_name="q_eval_model",
                    q_target_name="q_target_model",
                    replace=100
                    )
                # You can interleave training or train them sequentially
                steering_agent.train(num_episodes=100)
                speed_agent.train(num_episodes=100)

        elif choice == "4":
            # Test the controller
            run_notebook(os.path.join(os.getcwd(), 'src', 'game', 'controller.ipynb'))

        elif choice == "5":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()
