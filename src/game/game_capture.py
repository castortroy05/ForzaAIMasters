# Import the required libraries
from mss import mss
import cv2
import numpy as np
import pygetwindow as gw
import tensorflow as tf

# Configure the GPU to allow for memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU') 
    except RuntimeError as e:
        print(e)

class GameEnvironment:
    def __init__(self):
        """Initialize the game capture."""
        # Use mss to capture a screenshot of the game window
        self.sct = mss()
        window_title = "Forza Motorsport 7"
        game_window = gw.getWindowsWithTitle(window_title)
        if game_window:
            game_window = game_window[0]
            self.monitor = {
                "top": game_window.top,
                "left": game_window.left,
                "width": game_window.width,
                "height": game_window.height
            }
            # Store the expected shape based on the game window's dimensions
            self.expected_shape = (self.monitor['height'], self.monitor['width'], 3)
        else:
            raise ValueError("Could not find game window")

    def capture(self):
        """Capture the screen content of the game."""
        # Take a screenshot of the game window
        screenshot = self.sct.grab(self.monitor)
        # Convert the screenshot to a numpy array
        img = np.array(screenshot)
        # Check if the image was captured successfully
        if img is None or img.size == 0:
            raise Exception("Failed to capture screenshot or screenshot is empty")
        
        # Check if the image has the expected dimensions (e.g., height, width, channels)
        if len(img.shape) != 3 or img.shape[2] != 4:
            raise Exception(f"Unexpected image shape: {img.shape}")
        # Convert the image from RGBA to RGB
        img = img[:, :, :3]
        if img.shape != self.expected_shape:
            raise Exception(f"Unexpected image shape: {img.shape}")
        return img

    def speed_reward(self, position_red, position_blue, is_speed_up_colour, action):
        """
        Compute the reward based on game state and action taken.

        Parameters:
        - is_speed_up_colour (bool): Whether the speed up color is detected.
        - action (float): The action value taken by the agent.

        Returns:
        - reward (float): The computed reward.
        """
        # Base reward
        reward = 0.0

        # If the speed up color is detected, it means the agent should accelerate
        if is_speed_up_colour:
            if action > 0:  # If the agent is accelerating
                reward += 1.0
            else:  # If the agent is not accelerating or braking
                reward -= 1.0

        # If the speed up color is not detected, it means the agent should maintain or decelerate
        else:
            if action < 0:  # If the agent is braking or decelerating
                reward += 0.5
            elif action > 0:  # If the agent is still accelerating
                reward -= 0.5

        # Penalize extreme actions to encourage smoother adjustments
        if abs(action) > 0.9:
            reward -= 0.1

        return reward

    def steering_reward(self, position_red, position_blue):
        center = self.monitor['width'] // 2
        if position_red is not None:
            deviation = abs(position_red - center)
        elif position_blue is not None:
            deviation = abs(position_blue - center)
        else:
            return -5.0  # Increased penalty for not detecting any chevron

        # Reward is inversely proportional to the deviation from the center
        # We'll square the inverse to emphasize the importance of being close to the center
        reward = 1.0 / (1 + deviation**2)
        return reward

    def get_chevron_info(self, img):
        """Detect the chevron in the image and return its position and color."""
        templates = [cv2.imread(f'src/data/chevron{i}.png', cv2.IMREAD_COLOR) for i in range(1, 4)]
        if any(template is None for template in templates):
            print("Error: One or more template images not found.")
            return None, None
        speed_up_colour = templates[0][0, 0]
        img_copy = img.copy()  # Create a copy of the image
        img_copy = img_copy.astype(np.uint8)  # Ensure data type is uint8

        results = [cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) for template in templates]
        for result in results:
            print("Max match value:", np.max(result))
        threshold = 0.5
        locs = [np.where(result >= threshold) for result in results]
        loc = np.hstack(locs)
        if not loc[0].any() and not loc[1].any():
            return None, None, None
        for i, template in enumerate(templates):
            for pt in zip(*locs[i][::-1]):
                cv2.rectangle(img_copy, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0,0,255), 2)
        cv2.imshow('Detected Chevrons', img_copy)
        cv2.waitKey(1)  # Wait for 5 seconds
        cv2.destroyAllWindows()
        position_red = loc[1][0] + templates[0].shape[1] // 2
        position_blue = loc[1][-1] + templates[0].shape[1] // 2
        if position_red >= img.shape[0] or position_blue >= img.shape[1]:
            print("Warning: Detected position is out of image bounds.")
            return None, None, None
        colour_red = img[position_red, position_blue]
        colour_blue = img[position_red, position_blue]
        is_speed_up_colour = np.all(colour_red == speed_up_colour) or np.all(colour_blue == speed_up_colour)
        return position_red, position_blue, is_speed_up_colour

    def is_done(self, img):
        """Check if the game is over (e.g., car crash). Placeholder for now."""
        return False

    def draw_controller(self, img, action, position):
        """Draw a virtual controller on the image based on the action taken."""
        center_x, center_y = position
        cv2.circle(img, (center_x, center_y), 50, (255, 255, 255), 2)
        if action == 'accelerate':
            cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - 30), (0, 255, 0), 2)
        elif action == 'brake':
            cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y + 30), (0, 0, 255), 2)
        elif action == 'left':
            cv2.arrowedLine(img, (center_x, center_y), (center_x - 30, center_y), (255, 0, 0), 2)
        elif action == 'right':
            cv2.arrowedLine(img, (center_x, center_y), (center_x + 30, center_y), (255, 0, 0), 2)

    def display_overlays(self, predicted_speed_action=None, actual_speed_action=None, predicted_steering_action=None, actual_steering_action=None, position_red=None, position_blue=None):
        """Display only the overlays without the original image."""
        overlay_img = np.zeros((self.monitor['height'], self.monitor['width'], 3), dtype=np.uint8)
        self.draw_controller(overlay_img, actual_speed_action, position=(100, overlay_img.shape[0] - 100))
        self.draw_controller(overlay_img, actual_steering_action, position=(overlay_img.shape[1] - 100, overlay_img.shape[0] - 100))
        if predicted_speed_action is not None:
            # print(f"Debug: Predicted Speed Action = {predicted_speed_action}")
            cv2.putText(overlay_img, f"Predicted Speed Action: {predicted_speed_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if actual_speed_action is not None:
            cv2.putText(overlay_img, f"Actual Speed Action: {actual_speed_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if predicted_steering_action is not None:
            cv2.putText(overlay_img, f"Predicted Steering Action: {predicted_steering_action}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if actual_steering_action is not None:
            cv2.putText(overlay_img, f"Actual Steering Action: {actual_steering_action}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if position_red is not None:
            cv2.circle(overlay_img, (position_red, self.monitor['height']//2), 10, (0, 0, 255), 2)
        if position_blue is not None:
            cv2.circle(overlay_img, (position_blue, self.monitor['height']//2), 10, (255, 0, 0), 2)
        cv2.imshow("Overlays", overlay_img)
        cv2.waitKey(1)
