from game_env import GameEnvironment
from agents import SteeringAgent as SteeringAgent
from agents import SpeedAgent as SpeedAgent
from image_processing import preprocess_input_data
import threading
import traceback
import logging

screen_buffer = None
screen_ready = threading.Event()
game_env = GameEnvironment()

# Set up logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



def train_steering(agent, num_episodes, screen=None, detection_info=None):
    if screen is None or detection_info is None:
        screen = agent.game_env.capture()
        detection_info = agent.game_env.get_chevron_info(screen)
    state = preprocess_input_data(screen)
    for episode in range(num_episodes):
        state = agent.game_env.capture()
        state = preprocess_input_data(state)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            agent.simulate_action(action)
            next_state = agent.game_env.capture()
            chevron_info = agent.game_env.get_chevron_info(next_state)
            print(f"Chevron info: {chevron_info}")
            position_red, position_blue, is_speed_up_colour = chevron_info
            reward = agent.game_env.compute_steering_reward(position_red, position_blue)
            done = agent.game_env.is_done(next_state)  
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn(action)
            agent.game_env.display_overlays(next_state, steering_action=action, position_red=position_red, position_blue=position_blue)

        print(f'Steering - Episode: {episode}, Score: {score}')

def train_speed(agent, num_episodes, screen=None, detection_info=None):
    if screen is None or detection_info is None:
        screen = agent.game_env.capture()
        detection_info = agent.game_env.get_chevron_info(screen)
    state = preprocess_input_data(screen)
    for episode in range(num_episodes):
        state = agent.game_env.capture()
        state = preprocess_input_data(state)
        done = False
        score = 0
        action_name = 'accelerate'
        while not done:
            action = agent.choose_action(state)
            agent.simulate_action(action)
            next_state = agent.game_env.capture()
            chevron_info = agent.game_env.get_chevron_info(next_state)
            print(f"Chevron info: {chevron_info}")
            position_red, position_blue, is_speed_up_colour = chevron_info
            if action == 0:
                action_name = 'accelerate'
            elif action == 1:
                action_name = 'brake'
            reward = agent.game_env.compute_speed_reward(position_red, position_blue, is_speed_up_colour, action)
            print(f"Reward for action : {action}: {reward}")
            score += reward
            done = agent.game_env.is_done(next_state)  # Assuming you'll add is_done to GameEnvironment
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn(action)
            print(f"Predicted Speed Action: {action_name}, Actual Speed Action: {action_name}, Position Red: {position_red}, Position Blue: {position_blue}")
            agent.game_env.display_overlays(next_state, speed_action=action_name, position_red=position_red, position_blue=position_blue)
        print(f'Speed - Episode: {episode}, Score: {score}')

def train_concurrently(steering_agent, speed_agent, num_episodes=100):
    # Capture screen and perform detection
    screen = game_env.capture()
    logging.error(type(screen))
    logging.error(screen.shape)
    detection_info = game_env.get_chevron_info(screen)

    def wrapped_train_steering(agent, screen, detection_info, episodes):
        try:
            train_steering(agent, screen, detection_info, episodes)
        except Exception as e:
            error_msg = f"Error in train_steering: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())

    def wrapped_train_speed(agent, screen, detection_info, episodes):
        try:
            train_speed(agent, screen, detection_info, episodes)
        except Exception as e:
            error_msg = f"Error in train_speed: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())

    steering_thread = threading.Thread(target=wrapped_train_steering, args=(steering_agent, screen, detection_info, num_episodes))
    speed_thread = threading.Thread(target=wrapped_train_speed, args=(speed_agent, screen, detection_info, num_episodes))

    steering_thread.start()
    speed_thread.start()

    steering_thread.join()
    speed_thread.join()

    # Check if threads are alive
    if steering_thread.is_alive():
        warning_msg = "Warning: Steering thread is still running!"
        print(warning_msg)
        logging.warning(warning_msg)
    if speed_thread.is_alive():
        warning_msg = "Warning: Speed thread is still running!"
        print(warning_msg)
        logging.warning(warning_msg)