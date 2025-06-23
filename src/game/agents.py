import os
import numpy as np
from model import build_deep_learning_model
from tf_config import configure_tensorflow
from controller import GameController
from tensorflow.python.keras.models import load_model
from collections import deque
from learning import SteeringLearning, SpeedLearning

configure_tensorflow()

class BaseAgent:
    def __init__(self, game_env, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
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
        self.expected_shape = game_env.expected_shape

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.Q_eval.save_weights(os.path.join(path, "weights.h5"))

    def load_weights(self, path):
        if os.path.exists(path):
            self.Q_eval.load_weights(path)
        else:
            raise Exception("No such file exists")

    def learn(self, action):
        raise NotImplementedError("This method should be implemented in the derived class.")

    def train(self, num_episodes):
        raise NotImplementedError("This method should be implemented in the derived class.")

    def simulate_action(self, action):
        raise NotImplementedError("This method should be implemented in the derived class.")


class SteeringAgent(BaseAgent):
    def choose_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.uniform(-1, 1)
        else:
            state = np.array([observation])
            action = self.Q_eval.predict(state)[0][0]
        return action

    def learn(self, action):
        self.learning = SteeringLearning(
            self.Q_eval,
            self.Q_target,
            self.memory,
            self.gamma,
            self.replace,
            self.expected_shape,
            self.eps,
            self.eps_min,
            self.eps_dec,
        )
        self.learning.learn(action)

    def simulate_action(self, action):
        self.controller.simulate_steering_action(action)


class SpeedAgent(BaseAgent):
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
        self.learning = SpeedLearning(
            self.Q_eval,
            self.Q_target,
            self.memory,
            self.gamma,
            self.replace,
            self.expected_shape,
            self.eps,
            self.eps_min,
            self.eps_dec,
        )
        self.learning.learn(action)

    def simulate_action(self, action):
        self.controller.simulate_speed_action(action)
