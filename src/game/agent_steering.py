import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import vgamepad as vg
from game_capture import GameEnvironment

class Agent:
    def __init__(self, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
        self.Q_eval = self.build_deep_learning_model(input_dims, n_actions)
        self.Q_target = self.build_deep_learning_model(input_dims, n_actions)
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

    def build_deep_learning_model(self, input_dims, n_actions):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=input_dims, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(n_actions, activation=None))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def choose_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.Q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if len(self.memory) < 32:
            return
        if self.learn_step % self.replace == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())
        minibatch = random.sample(self.memory, 32)
        state, action, reward, new_state, done = zip(*minibatch)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        new_state = np.array(new_state)
        done = np.array(done)
        action_values = self.Q_eval.predict(state)
        next_action_values = self.Q_target.predict(new_state)
        max_next_action_values = np.max(next_action_values, axis=1)
        Q_target = action_values.copy()
        batch_index = np.arange(32, dtype=int)
        Q_target[batch_index, action] = reward + self.gamma * max_next_action_values * ~done
        self.Q_eval.train_on_batch(state, Q_target)
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec
        self.learn_step += 1

    def train(self, num_episodes):
        game_env = GameEnvironment()
        for episode in range(num_episodes):
            state = game_env.capture()
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
                self.learn()
                if action == 0:
                    print("Action chosen: Left")
                elif action == 1:
                    print("Action chosen: Right")
            print(f'Episode: {episode}, Score: {score}')
    try:
        self.Q_eval.save('model.h5')
    except Exception as e:
        print(e)



    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.Q_eval.save_weights(os.path.join(path, "weights.h5"))

    def load_weights(self, path):
        if os.path.exists(path):
            self.Q_eval.load_weights(path)
        else:
            raise Exception("No such file exists")

    def simulate_action(self, action):
        if action == 0:
            self.controller.left_stick_x = -1.0  # Simulate left turn
        elif action == 1:
            self.controller.left_stick_x = 1.0   # Simulate right turn
        self.controller.update()
