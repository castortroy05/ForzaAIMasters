import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import vgamepad as vg
from game_capture import GameEnvironment

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for the GPUs to be used
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set GPU to be used
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')  # Use the second GPU
    except RuntimeError as e:
        print(e)
        
class Agent:
    def __init__(self, input_dims, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_target_name, replace):
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
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=input_dims, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))
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
    
    def preprocess_input_data(self, img):
        # Extract relevant features from the image (e.g., color histograms, edge detection, etc.)
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
        # Add additional features to match the expected input shape of the model
        # features.extend([0, 0, 0, 0])
        while len(features) < 6:
            features.append(0)
        return np.array(features)



    def learn(self, action):
        if len(self.memory) < 32:
            return
        if self.learn_step % self.replace == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())
        minibatch = random.sample(self.memory, 32)
        state, action, reward, new_state, done = zip(*minibatch)
        # for s in state:
        #     print(f"Shape of element in state speed: {s.shape}")
        # Filter out elements with unexpected shapes
        state = [s for s in state if s.shape == (634, 1067, 3)]
        state = np.array([self.preprocess_input_data(s) for s in state])
        action = np.array(action)
        reward = np.array(reward)
        new_state = np.array([self.preprocess_input_data(s) for s in new_state])
        done = np.array(done)
        # Filter out elements with unexpected shapes
        valid_indices = [i for i, s in enumerate(state) if s.shape == (634, 1067, 3)]
        state = [state[i] for i in valid_indices]
        state = np.array(state)
        action = [action[i] for i in valid_indices]
        action = np.array(action)
        reward = [reward[i] for i in valid_indices]
        reward = np.array(reward)
        new_state = [new_state[i] for i in valid_indices]
        new_state = np.array(new_state)
        done = [done[i] for i in valid_indices]
        done = np.array(done)
        if state.size == 0:
            print("State is empty. Skipping speed prediction.")
            return
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
            state = self.game_env.capture()
            state = self.preprocess_input_data(state)
            done = False
            score = 0
            action_name = 'accelerate'
            while not done:
                action = self.choose_action(state)
                self.simulate_action(action)
                next_state = self.game_env.capture()
                position_red, position_blue, is_speed_up_colour = self.game_env.get_chevron_info(next_state)
                if action == 0:
                    action_name = 'accelerate'
                elif action == 1:
                    action_name = 'brake'
                reward = self.game_env.speed_reward(position_red, position_blue, is_speed_up_colour, action_name)
                print(f"Action Choice: {action_name}" + " Reward: " + str(reward))
                score += reward
                done = self.game_env.is_done(next_state)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                self.learn(action)  # Pass the chosen action to the learn function

                # Pass the chosen actions to the display method
                game_env.display(next_state, predicted_speed_action=action_name, actual_speed_action=action_name, position_red=position_red, position_blue=position_blue)
            print(f'Episode: {episode}, Score: {score}')


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
        # action should be a value between -1 and 1 representing the degree of acceleration or braking
        if action >= 0:
            self.controller.right_trigger(value=int(action * 255))  # Simulate acceleration
        else:
            self.controller.left_trigger(value=int(-action * 255))   # Simulate braking
        self.controller.update()

