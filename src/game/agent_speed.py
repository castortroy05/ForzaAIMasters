# Import necessary libraries and modules
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from collections import deque
import random
import vgamepad as vg
from game_capture import GameEnvironment
game_env = GameEnvironment()

# Check for available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for the GPUs to be used
        # This allows TensorFlow to allocate GPU memory based on runtime requirements
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set the second GPU to be used by TensorFlow
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        # Print any errors that occur during GPU configuration
        print(e)

        
# Define the Agent class responsible for training and decision-making
class Agent:
    def __init__(self, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
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
        self.Q_eval = self.build_deep_learning_model(input_dims)
        self.Q_target = self.build_deep_learning_model(input_dims)
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
        self.controller = vg.VX360Gamepad()
        self.game_env = GameEnvironment()


    def build_deep_learning_model(self, input_dims):
        """
        Construct a deep learning model for Q-learning.

        Parameters:
        - input_dims (int): Number of input dimensions or features for the model.

        Returns:
        - model (tf.keras.models.Sequential): The constructed deep learning model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=input_dims, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

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
        """
        Choose an action based on the current state using the epsilon-greedy strategy.

        Parameters:
        - observation (array-like): The current state.

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
        Extract relevant features from the image, such as mean and standard deviation for each color channel.

        Parameters:
        - img (array-like): The input image.

        Returns:
        - features (np.array): Extracted features from the image.
        """
        features = []
        if len(img.shape) == 3:
            for i in range(3):
                features.append(np.mean(img[:,:,i]))
                features.append(np.std(img[:,:,i]))
        elif len(img.shape) == 1:
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
        # If there are fewer than 32 experiences in memory, return without learning
        if len(self.memory) < 32:
            return

        # Update the Q-target weights with Q-evaluation weights at specified intervals
        if self.learn_step % self.replace == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())

        # Sample a minibatch of experiences from memory
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
        batch_index = np.arange(32, dtype=int)
        Q_target[batch_index, action] = reward + self.gamma * max_next_action_values * ~done

        # Train the Q-evaluation network
        self.Q_eval.train_on_batch(state, Q_target)

        # Decay the epsilon value
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec

        # Increment the learning step
        self.learn_step += 1

    def train(self, num_episodes):
        """
        Train the agent for a specified number of episodes.

        Parameters:
        - num_episodes (int): The number of episodes to train the agent for.
        """
        for episode in range(num_episodes):
            # Capture the game state
            state = self.game_env.capture()
            state = self.preprocess_input_data(state)
            done = False
            score = 0
            action_name = 'accelerate'

            # Continue taking actions and learning until the episode ends
            while not done:
                action = self.choose_action(state)
                self.simulate_action(action)
                next_state = self.game_env.capture()

                # Compute the reward based on game state and action taken
                position_red, position_blue, is_speed_up_colour = self.game_env.get_chevron_info(next_state)
                if action == 0:
                    action_name = 'accelerate'
                elif action == 1:
                    action_name = 'brake'
                reward = self.game_env.speed_reward(position_red, position_blue, is_speed_up_colour, action_name)

                # Update the score and store the experience
                score += reward
                done = self.game_env.is_done(next_state)
                self.store_transition(state, action, reward, next_state, done)

                # Move to the next state and learn from the experience
                state = next_state
                self.learn(action)

                # Display the game state and actions
                game_env.display(next_state, predicted_speed_action=action_name, actual_speed_action=action_name, position_red=position_red, position_blue=position_blue)

            # Print the episode's score
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
        # If the action is positive, simulate acceleration
        if action >= 0:
            self.controller.right_trigger(value=int(action * 255))
        # If the action is negative, simulate braking
        else:
            self.controller.left_trigger(value=int(-action * 255))
        self.controller.update()