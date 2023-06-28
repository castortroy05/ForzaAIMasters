# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import vgamepad as vg

# Defining the Agent class
class Agent:
    # Initializing the Agent object with the given parameters
    def __init__(self, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
        # Building two deep learning models for Q-evaluation and Q-target
        self.Q_eval = self.build_deep_learning_model(input_dims, n_actions)
        self.Q_target = self.build_deep_learning_model(input_dims, n_actions)
        # Initializing memory with a maximum size of mem_size
        self.memory = deque(maxlen=mem_size)
        # Setting the initial value of epsilon for exploration-exploitation trade-off
        self.eps = eps
        # Setting the minimum value of epsilon
        self.eps_min = eps_min
        # Setting the rate at which epsilon will decrease over time
        self.eps_dec = eps_dec
        # Setting the discount factor for future rewards
        self.gamma = gamma
        # Setting the frequency at which the target network will be updated
        self.replace = replace
        # Creating an action space with n_actions discrete actions
        self.action_space = [i for i in range(n_actions)]
        # Initializing the learn step counter to 0
        self.learn_step = 0
        # Setting the names for the Q-evaluation and Q-target models for saving and loading weights
        self.q_eval_name = q_eval_name
        self.q_target_name = q_target_name
        # Initializing a virtual gamepad controller using vgamepad library
        self.controller = vg.VX360Gamepad()

    # Defining a method to build a deep learning model with two hidden layers and an output layer with n_actions neurons
    def build_deep_learning_model(self, input_dims, n_actions):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=input_dims, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(n_actions, activation=None))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    # Defining a method to train the deep learning model
    def train(self):
        # Defining the batch size
        batch_size = 64
        # Checking if the deep learning model is not empty
        if self.Q_eval is not None:
            # Checking if the length of the memory is greater than the batch size
            if len(self.memory) > batch_size:
                # Getting a random sample of transitions from the memory
                sample = random.sample(self.memory, batch_size)
                # Getting the states, actions, rewards, and new states from the random sample
                states = np.array([transition[0] for transition in sample])
                actions = np.array([transition[1] for transition in sample])
                rewards = np.array([transition[2] for transition in sample])
                new_states = np.array([transition[3] for transition in sample])
                # Predicting the Q values of the states using the deep learning model
                q_values = self.Q_eval.predict(states)
                # Predicting the Q values of the new states using the deep learning model
                new_q_values = self.Q_eval.predict(new_states)
                # Looping through the sample
                for index in range(len(sample)):
                    # Checking if the done flag is true
                    if sample[index][4]:
                        # Updating the Q values of the actions
                        q_values[index][actions[index]] = rewards[index]
                    else:
                        # Updating the Q values of the actions
                        q_values[index][actions[index]] = rewards[index] + self.gamma * np.max(new_q_values[index])
                # Training the deep learning model with the updated Q values of the actions
                self.Q_eval.fit(states, q_values, epochs=1, verbose=0)


    # Defining a method to store a transition in memory
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    # Defining a method to choose an action based on the current observation and epsilon-greedy policy
    def choose_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.Q_eval.predict(state)
            action = np.argmax(actions)
        return action

    # Defining a method to learn from a minibatch of transitions sampled from memory
    def learn(self):
        if len(self.memory) < 32:
            return
        if self.learn_step % self.replace == 0:
                    # Unpacking the minibatch into separate arrays for each element of the transition tuple
                    
            self.Q_target.set_weights(self.Q_eval.get_weights())
        minibatch = random.sample(self.memory, 32)
        state, action, reward, new_state, done = zip(*minibatch)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        new_state = np.array(new_state)
        done = np.array(done)
        # Predicting the Q-values for the current state using the Q-evaluation model
        action_values = self.Q_eval.predict(state)
        # Predicting the Q-values for the next state using the Q-target model
        next_action_values = self.Q_target.predict(new_state)
        # Selecting the maximum Q-value for each next state
        max_next_action_values = np.max(next_action_values, axis=1)
        # Creating a copy of the predicted Q-values for the current state to update them
        Q_target = action_values.copy()
        # Creating an array of indices for the minibatch
        batch_index = np.arange(32, dtype=np.int32)
        # Updating the Q-value for the selected action using the Bellman equation
        Q_target[batch_index, action] = reward + self.gamma * max_next_action_values * ~done
        # Training the Q-evaluation model on the current state and updated Q-values
        self.Q_eval.train_on_batch(state, Q_target)
        # Decreasing epsilon if it is greater than its minimum value
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec
        # Incrementing the learn step counter
        self.learn_step += 1


def train(self, num_episodes):
    """
    Train the agent for a specified number of episodes.
    """
    for episode in range(num_episodes):
        # Reset the game environment and get the initial state
        state = self.env.reset()
        done = False
        score = 0

        while not done:
            # Choose an action
            action = self.choose_action(state)
            # Take the action and get the new state and reward
            next_state, reward, done, info = self.env.step(action)
            # Update the score
            score += reward
            # Store the experience in memory
            self.remember(state, action, reward, next_state, done)
            # Make the next state the current state
            state = next_state
            # Learn from the experience
            self.learn()

        # Print the score for this episode
        print(f'Episode: {episode}, Score: {score}')

    try:
        # Save the trained model
        self.model.save('model.h5')
    except Exception as e:
        print(e)

    def save_weights(self, path):
        if not os.path.isdir(path):
            raise ValueError("The path must be a directory")
        self.model.save_weights(os.path.join(path, "weights.h5"))

    # Defining a method to load weights from a file into the deep learning model
    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_weights(path)
        else:
            raise Exception("No such file exists")
