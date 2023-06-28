# Autonomous Racing Game AI

This project aims to develop an AI that can autonomously play a racing game. The AI uses reinforcement learning to learn how to control a car in the game environment. The project is divided into several components, each with its own responsibilities.

## Project Structure

The project is structured as follows:

- `agent_steering.py`: This file defines the agent responsible for steering the car. The agent uses a deep Q-learning network to learn the optimal steering actions based on the game state.

- `agent_speed.py`: This file defines the agent responsible for controlling the speed of the car. Like the steering agent, the speed agent also uses a deep Q-learning network to learn the optimal speed actions.

- `game_capture.py`: This file contains the code for capturing the game screen. The captured screen is used as the game state, which is fed into the agents to decide the next action.

- `play.py`: This file contains the main game loop. It initializes the game environment and the agents, and then runs the game loop where the agents interact with the game environment.

- `utils.py`: This file contains utility functions used throughout the project, such as functions for saving and loading models, and for preprocessing and postprocessing game data.

## How to Run the Project

1. Ensure that you have all the necessary dependencies installed. These include TensorFlow, OpenCV, MSS, and vgamepad.

2. Run the `play.py` file to start the game loop. This will initialize the game environment and the agents, and start the game.

3. The agents will start interacting with the game environment, and you should see the car being controlled by the AI.

## Training the Agents

The agents are trained using reinforcement learning. They learn by interacting with the game environment and receiving rewards based on their actions. The agents aim to maximize their cumulative reward over time.

The agents' memories of past experiences are stored in a replay buffer. During training, the agents sample experiences from this buffer and learn from them. This approach, known as experience replay, helps to stabilize the learning process.

## Future Work

There are several ways in which this project could be extended in the future:

- Implement more sophisticated reward functions to encourage the agents to learn more complex behaviors.

- Use more advanced reinforcement learning algorithms, such as actor-critic methods or policy gradients.

- Incorporate additional game features into the state representation, such as the position of other cars or the layout of the track.

- Train the agents on a variety of different tracks to increase their generalization ability.

- Implement a user interface to allow for easy configuration of the agents and the game environment.
