# Import necessary libraries and modules
import sys
import os

# Set TensorFlow log level to show full logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf

# Uncomment the following line if you want to specify which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add the 'game' directory to the system path for importing game-related modules
sys.path.append(os.path.join(os.getcwd(), 'src/game/'))

# Import agents responsible for steering and speed control
import agent_steering
import agent_speed

# Import libraries for window management and Jupyter notebook execution
import pygetwindow as gw
import nbformat
from IPython.core.interactiveshell import InteractiveShell

def is_game_running(window_title):
    """
    Check if a specific game window is currently running.

    Parameters:
    - window_title (str): The title of the game window to check.

    Returns:
    - bool: True if the game window is running, False otherwise.
    """
    try:
        windows = gw.getWindowsWithTitle(window_title)
        return len(windows) > 0
    except Exception:
        return False

def run_notebook(notebook_path):
    """
    Execute a Jupyter notebook programmatically.

    Parameters:
    - notebook_path (str): The path to the Jupyter notebook to run.
    """
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    shell = InteractiveShell.instance()
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            shell.run_cell(cell.source)

def main():
    """
    Main function to train reinforcement learning agents for game control.
    """
    while True:
        # Clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Display the welcome message and instructions
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

        # Get user input for the desired action
        choice = input("Enter your choice (1/2/3/4/5): ")

        # Check if the game is running before training
        if choice in ["1", "2", "3"]:
            if not is_game_running("Forza Motorsport 7"):
                print("The game is not currently running. Please start the game before training.")
                continue

            # Train the steering agent
        if choice == "1":
            # Initialize the steering agent with the following parameters:
            # - input_dims: The number of input dimensions or features the agent expects.
            # - n_actions: The number of possible actions the agent can take.
            # - mem_size: The size of the agent's memory for storing experiences.
            # - eps: Initial value for epsilon in epsilon-greedy action selection. 
            #        It represents the probability of the agent taking a random action.
            # - eps_min: The minimum value that epsilon can decay to.
            # - eps_dec: The decay rate of epsilon after each episode.
            # - gamma: The discount factor used in the Q-learning update rule.
            # - q_eval_name: Name of the evaluation Q-network model.
            # - q_target_name: Name of the target Q-network model.
            # - replace: Frequency (in terms of steps) of replacing target network with evaluation network.
            steering_agent = agent_steering.Agent(
                input_dims=6,
                n_actions=2, 
                mem_size=1000,
                eps=1.0,
                eps_min=0.01,
                eps_dec=0.995,
                gamma=0.99,
                q_eval_name="q_eval_model",
                q_target_name="q_target_model",
                replace=100
            )
            # Train the steering agent for a specified number of episodes.
            steering_agent.train(num_episodes=100)

        # Train the speed agent
        elif choice == "2":
            # Initialize the speed agent with parameters similar to the steering agent.
            speed_agent = agent_speed.Agent(
                input_dims=6,
                n_actions=2, 
                mem_size=1000,
                eps=1.0,
                eps_min=0.01,
                eps_dec=0.995,
                gamma=0.99,
                q_eval_name="q_eval_model",
                q_target_name="q_target_model",
                replace=100
            )
            # Train the speed agent for a specified number of episodes.
            speed_agent.train(num_episodes=100)

        # Train both agents
        elif choice == "3":
            # Initialize both the steering and speed agents with parameters similar to above.
            steering_agent = agent_steering.Agent(
                input_dims=6,
                n_actions=2, 
                mem_size=1000,
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
                n_actions=2, 
                mem_size=1000,
                eps=1.0,
                eps_min=0.01,
                eps_dec=0.995,
                gamma=0.99,
                q_eval_name="q_eval_model",
                q_target_name="q_target_model",
                replace=100
            )
            # Train both agents for a specified number of episodes.
            steering_agent.train(num_episodes=100)
            speed_agent.train(num_episodes=100)


        # Test the controller by running the controller notebook
        elif choice == "4":
            run_notebook(os.path.join(os.getcwd(), 'src', 'game', 'controller.ipynb'))

        # Exit the program
        elif choice == "5":
            print("Exiting the program. Goodbye!")
            break

        # Handle invalid choices
        else:
            print("Invalid choice. Please choose a valid option.")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
