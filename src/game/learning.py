from image_processing import preprocess_input_data
import numpy as np
import random

class BaseLearning:
    def __init__(self, Q_eval, Q_target, memory, gamma, replace, expected_shape,
                 eps, eps_min, eps_dec):
        self.Q_eval = Q_eval
        self.Q_target = Q_target
        self.memory = memory
        self.gamma = gamma
        self.replace = replace
        self.expected_shape = expected_shape
        self.learn_step = 0
        # Epsilon parameters used in the learning loop
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec

    def preprocess_input_data(self, img):
        return preprocess_input_data(img)

    def learn(self, action):
        raise NotImplementedError("This method should be implemented in the derived class.")


class SteeringLearning(BaseLearning):
    def learn(self, action):
        # The specific learning logic for the steering agent
        if len(self.memory) < 32:
            return
        if self.learn_step % self.replace == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())
        minibatch = random.sample(self.memory, 32)
        state, action, reward, new_state, done = zip(*minibatch)
        state = np.array([self.preprocess_input_data(s) for s in state])
        new_state = np.array([self.preprocess_input_data(s) for s in new_state])
        
        # Predict Q-values for current states and next states
        action_values = self.Q_eval.predict(state)
        next_action_values = self.Q_target.predict(new_state)

        # Filter out elements with unexpected shapes
        state = [s for s in state if s.shape == self.expected_shape]
        state = np.array([self.preprocess_input_data(s) for s in state])
        action = np.array(action)
        reward = np.array(reward)
        new_state = np.array([self.preprocess_input_data(s) for s in new_state])
        done = np.array(done)
        # Filter out elements with unexpected shapes
        valid_indices = [i for i, s in enumerate(state) if s.shape == self.expected_shape]
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
            print("State is empty. Skipping steering prediction.")
            return

        # Compute the target Q-values
        # max_next_action_values = np.max(next_action_values, axis=1)
        # Q_target = action_values.copy()
        # batch_index = np.arange(Q_target.shape[0])
        # Q_target[batch_index, action] = reward + self.gamma * max_next_action_values * ~done
        Q_target = reward + self.gamma * next_action_values * ~done

        # Train the Q-evaluation network
        self.Q_eval.train_on_batch(state, Q_target)
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec
        self.learn_step += 1
        


class SpeedLearning(BaseLearning):
    def learn(self, action):
        # The specific learning logic for the speed agent
        if len(self.memory) < 32:
            return
        if self.learn_step % self.replace == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())
        minibatch = random.sample(self.memory, 32)
        state, action, reward, new_state, done = zip(*minibatch)
        state = np.array([self.preprocess_input_data(s) for s in state])
        new_state = np.array([self.preprocess_input_data(s) for s in new_state])
        # Filter out elements with unexpected shapes
        state = [s for s in state if s.shape == self.expected_shape]
        state = np.array([self.preprocess_input_data(s) for s in state])
        action = np.array(action)
        reward = np.array(reward)
        new_state = np.array([self.preprocess_input_data(s) for s in new_state])
        done = np.array(done)
        # Filter out elements with unexpected shapes
        valid_indices = [i for i, s in enumerate(state) if s.shape == self.expected_shape]
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
        # Predict Q-values for current states and next states
        action_values = self.Q_eval.predict(state)
        next_action_values = self.Q_target.predict(new_state)

        # Compute the target Q-values
        max_next_action_values = np.amax(next_action_values, axis=1)
        Q_target = action_values.copy()
        batch_index = np.arange(32, dtype=int)
        Q_target[batch_index, action] = reward + self.gamma * max_next_action_values * ~done
        print(f"Q-values before learning: {action_values}")
        print(f"Target Q-values: {Q_target}")
        # Train the Q-evaluation network
        self.Q_eval.train_on_batch(state, Q_target)

        # Decay the epsilon value
        if self.eps > self.eps_min:
            self.eps *= self.eps_dec

        # Increment the learning step
        self.learn_step += 1
