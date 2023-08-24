import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from collections import deque
import random
import vgamepad as vg
from game_capture import GameEnvironment

# List all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

# Enable eager execution for TensorFlow operations
tf.config.run_functions_eagerly(True)

# Enable debug mode for TensorFlow data operations
tf.data.experimental.enable_debug_mode()

# Check if there are available GPUs
if gpus:
    try:
        # Set memory growth for each GPU to avoid memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Explicitly set TensorFlow to use the second GPU
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

class Agent:
    def __init__(self, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
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
        self.Q_eval = self.build_deep_learning_model(input_dims)
        self.Q_eval.compile(optimizer=Adam(), loss='mse', run_eagerly=True)
        print(f"self.Q_eval.summary(): {self.Q_eval.summary()}")
        self.Q_target = self.build_deep_learning_model(input_dims)

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
        self.controller = vg.VX360Gamepad()

    def build_deep_learning_model(self, input_dims):
        """
        Build a deep learning model for Q-value approximation.

        Parameters:
        - input_dims (int): The number of input dimensions for the model.

        Returns:
        - model (tf.keras.models.Sequential): The deep learning model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=input_dims, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

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
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.Q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def preprocess_input_data(self, img):
        """
        Preprocess the input image data to extract relevant features.

        Parameters:
        - img (array-like): The input image.

        Returns:
        - features (np.array): The extracted features.
        """
        features = []
        if len(img.shape) == 3:  # If the img array is 3-dimensional
            for i in range(3):  # For each color channel (R, G, B)
                features.append(np.mean(img[:,:,i]))
                features.append(np.std(img[:,:,i]))
        elif len(img.shape) == 1:  # If the img array is 1-dimensional
            features.append(np.mean(img))
            features.append(np.std(img))
        else:
            raise ValueError(f"Unexpected shape of img array: {img.shape}")

        while len(features) < 6:
            features.append(0)
        return np.array(features)

    def learn(self, action):
        """
        Update the Q-values using experiences from the memory.

        Parameters:
        - action (int): The action taken by the agent.
        """
        if len(self.memory) < 32:
            return
        if self.learn_step % self.replace == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())
        minibatch = random.sample(self.memory, 32)
        state, action, reward, new_state, done = zip(*minibatch)

        # Preprocess the states and new states
        state = np.array([self.preprocess_input_data(s) for s in state])
        new_state = np.array([self.preprocess_input_data(s) for s in new_state])

        # Predict Q-values for current states and next states
        action_values = self.Q_eval.predict(state)
        next_action_values = self.Q_target.predict(new_state)

        # Compute the target Q-values
        max_next_action_values = np.max(next_action_values, axis=1)
        Q_target = action_values.copy()
        batch_index = np.arange(Q_target.shape[0])
        Q_target[batch_index, action] = reward + self.gamma * max_next_action_values * ~done

        # Train the Q-evaluation network
        self.Q_eval.train_on_batch(state, Q_target)
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec
        self.learn_step += 1

    def train(self, num_episodes):
        """
        Train the agent for a specified number of episodes.

        Parameters:
        - num_episodes (int): The number of episodes to train the agent for.
        """
        game_env = GameEnvironment()
        for episode in range(num_episodes):
            state = game_env.capture()
            state = self.preprocess_input_data(state)
            done = False
            score = 0
            while not done:
                action = self.choose_action(state)
                self.simulate_action(action)
                next_state = game_env.capture()
                position_red, position_blue = game_env.get_chevron_info(next_state)
                reward = game_env.steering_reward(position_red, position_blue)
                done = game_env.is_done(next_state)
                score += reward
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                self.learn(action)  # Pass the chosen action to the learn function

                # Pass the chosen actions to the display method
                game_env.display(next_state, predicted_steering_action=action, actual_steering_action=action, position_red=position_red, position_blue=position_blue)
            print(f'Episode: {episode}, Score: {score}')

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
        x_value = int(action * 32767)
        self.controller.left_joystick(x_value=x_value, y_value=0)
        self.controller.update()
