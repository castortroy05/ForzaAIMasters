import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from vgamepad import VG360Gamepad

class AgentSpeed:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.model = self.build_model()
        self.memory = SequentialMemory(limit=50000, window_length=1)
        self.policy = BoltzmannQPolicy()
        self.dqn = self.build_agent(self.model, self.actions)
        self.gamepad = VG360Gamepad()

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.states))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        return model

    def build_agent(self, model, actions):
        dqn = DQNAgent(model=model, memory=self.memory, policy=self.policy,
                       nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
        return dqn

    def choose_action(self, state):
        action = self.dqn.forward(state)
        if action == 0:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self.gamepad.update()
        elif action == 1:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
        return action

def learn(self):
    if len(self.memory) < self.batch_size:
        return

    # Sample a batch of experiences from the agent's memory
    mini_batch = random.sample(self.memory, self.batch_size)

    # Extract the states, actions, rewards, new states and done flags from the mini batch
    states, actions, rewards, new_states, done_flags = self.extract_from_batch(mini_batch)

    # Predict the Q-values of the new states
    future_qs_list = self.model.predict(new_states)

    # Initialize an empty array for the targets
    X = []
    y = []

    # Compute the target Q-values
    for index, (state, action, reward, new_state, done) in enumerate(mini_batch):
        if not done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.gamma * max_future_q
        else:
            new_q = reward

        # Update the Q-value for the action
        current_qs = self.model.predict(state)
        current_qs[0][action] = new_q

        # Add the state and the updated Q-values to the training data
        X.append(state)
        y.append(current_qs[0])

    # Reshape the training data
    X = np.array(X).reshape(-1, *self.state_shape)
    y = np.array(y).reshape(-1, self.action_space)

    # Train the model on the training data
    self.model.fit(X, y, batch_size=self.batch_size, verbose=0, shuffle=False)

    # Update the exploration rate
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

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

    # Save the trained model
    self.model.save('model.h5')

def save(self, path):
        self.dqn.save_weights(path)

def load(self, path):
        self.dqn.load_weights(path)