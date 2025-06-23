import os
import numpy as np
from model import build_deep_learning_model
from tf_config import configure_tensorflow
from controller import GameController
from tensorflow.python.keras.models import load_model
from collections import deque
from learning import SteeringLearning
# The training module exposes a function named `train_steering`.
# Import that function instead of the non-existent `train_steering_agent`.
from training import train_steering

class Agent:
    def __init__(self, game_env, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
        """
        Initialize the Agent.

        Parameters:
        - input_dims (int): The number of input dimensions for the deep learning model.
        - n_actions (int): The number of possible actions the agent can take.
        - mem_size (int): Maximum size of the memory buffer.
        - eps (float): Initial epsilon value for epsilon-greedy action selection.
        - eps_min (float): Minimum epsilon value.
        - eps_dec (float): Decay rate for epsilon.
        - gamma (float): Discount factor for future rewards.
        - q_eval_name (str): Name for the Q-evaluation model.
        - q_target_name (str): Name for the Q-target model.
        - replace (int): Frequency of replacing Q-target weights with Q-evaluation weights.
        """
        # Build the Q-evaluation and Q-target deep learning models
        self.Q_eval = build_deep_learning_model(input_dims)
        # self.Q_eval.compile(optimizer=Adam(), loss='mse', run_eagerly=True)
        print(f"self.Q_eval.summary(): {self.Q_eval.summary()}")
        self.Q_target = build_deep_learning_model(input_dims)
        self.game_env = game_env
        self.expected_shape =game_env.expected_shape

        # Initialize memory buffer, epsilon values, gamma, replace frequency, and learning step counter
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

        # Initialize the game controller
        self.controller = GameController()

    
    def store_transition(self, state, action, reward, new_state, done):
        """
        Store a transition (experience) in the memory buffer.

        Parameters:
        - state (array-like): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - new_state (array-like): The next state.
        - done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, new_state, done))

    def choose_action(self, observation):
        """
        Choose an action based on the current observation using epsilon-greedy policy.

        Parameters:
        - observation (array-like): The current observation.

        Returns:
        - action (int): The chosen action.
        """
        if np.random.random() < self.eps:
            action = np.random.uniform(-1, 1)  # Return continuous value between -1 and 1
        else:
            state = np.array([observation])
            action = self.Q_eval.predict(state)[0][0]  # Directly get the predicted action value
        return action

    def learn(self, action):
        self.learning.learn(action)
    
    def train(self, num_episodes):
        train_steering(self, num_episodes)
        
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
        # action should be a value between -1 and 1 representing the degree of steering
        self.controller.simulate_steering_action(action)
