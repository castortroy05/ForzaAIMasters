# Import necessary libraries and modules
import os
import numpy as np
from tensorflow.python.keras.models import load_model
from model import build_deep_learning_model
from tf_config import configure_tensorflow
from controller import GameController
from collections import deque
import random
from learning import SpeedLearning
# The correct training function is named `train_speed`. Import it
# instead of the non-existent `train_speed_agent`.
from training import train_speed

from game_capture import GameEnvironment
        
# Define the Agent class responsible for training and decision-making
class Agent:
    def __init__(self, game_env, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
        """
        Initialize the agent.

        Parameters:
        - input_dims (int): Number of input dimensions or features for the model.
        - n_actions (int): Number of possible actions the agent can take.
        - mem_size (int): Maximum number of experiences the agent can store.
        - eps (float): Initial probability of taking a random action (for epsilon-greedy strategy).
        - eps_min (float): Minimum value that epsilon can decay to.
        - eps_dec (float): Decay rate for epsilon after each episode.
        - gamma (float): Discount factor for future rewards.
        - q_eval_name (str): Name for the Q-evaluation model.
        - q_target_name (str): Name for the Q-target model.
        - replace (int): Frequency to replace target network weights with evaluation network weights.
        """
        configure_tensorflow()
        self.Q_eval = build_deep_learning_model(input_dims)
        self.Q_target = build_deep_learning_model(input_dims)
        self.memory = deque(maxlen=mem_size)
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.gamma = gamma
        self.replace = replace
        self.action_space = [i for i in range(n_actions)]
        self.learn_step = 0
        self.q_eval_name = q_eval_name
        self.q_target_name = q_target_name
        self.controller = GameController()
        self.game_env = game_env
        self.expected_shape =game_env.expected_shape

    def store_transition(self, state, action, reward, new_state, done):
        """
        Store an experience in the agent's memory.

        Parameters:
        - state (array-like): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - new_state (array-like): The state after taking the action.
        - done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, new_state, done))


    def choose_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.choice(self.action_space)
            print(f"Random action: {action}")
            return action
        
        state = np.array([observation])
        q_values = self.Q_eval.predict(state)
        chosen_action = np.argmax(q_values[0])
        print(f"Q-values: {q_values[0]}, Chosen action: {chosen_action}")
        return chosen_action


    
 



    def learn(self, action):
        self.learning.learn(action)

    def train(self, num_episodes):
        """
        Train the agent for a specified number of episodes.

        Parameters:
        - num_episodes (int): The number of episodes to train the agent for.
        """
        train_speed(self, num_episodes)

    def save_weights(self, path):
        """
        Save the Q-evaluation network weights to a file.

        Parameters:
        - path (str): The directory path to save the weights.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.Q_eval.save_weights(os.path.join(path, "weights.h5"))

    def load_weights(self, path):
        """
        Load the Q-evaluation network weights from a file.

        Parameters:
        - path (str): The file path to load the weights from.
        """
        if os.path.exists(path):
            self.Q_eval.load_weights(path)
        else:
            raise Exception("No such file exists")

    def simulate_action(self, action):
        """
        Simulate the chosen action in the game environment.

        Parameters:
        - action (int): The action to simulate.
        """
        self.controller.simulate_speed_action(action)