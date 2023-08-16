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

    def __init__(self):
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
            else:
                raise ValueError("Could not find game window")


    def capture(self):
            """Capture the screen"""
            # Take a screenshot of the whole screen
            screenshot = self.sct.grab(self.monitor)

            # Convert the screenshot to a numpy array
            img = np.array(screenshot)

            # Check that the image was captured successfully
            if img is None:
                raise Exception("Failed to capture screenshot")

            # Convert the image from RGBA to RGB
            img = img[:, :, :3]  # Convert from RGBA to RGB by slicing the image array

            # Return the image
            return img
    
    def speed_reward(self, position_red, position_blue, is_speed_up_colour, action):
        # If the speed up colour is default
        if is_speed_up_colour:
            # If the chosen action is 'accelerate', return a positive reward
            if action == 'accelerate':
                return 1.0
            # Otherwise, return a negative reward
            elif action == 'brake':
                return -1.0
            else:
                raise ValueError('Unknown action: {}'.format(action))
        # If the speed up colour is red
        else:
            # If the chosen action is 'brake', return a positive reward
            if action == 'brake':
                return 1.0
            # Otherwise, return a negative reward
            elif action == 'accelerate':
                return -1.0
            else:
                raise ValueError('Unknown action: {}'.format(action))


    def steering_reward(self, position_red, position_blue):
        center = self.monitor['width'] // 2
        if position_red is not None:
            deviation = abs(position_red - center)
        elif position_blue is not None:
            deviation = abs(position_blue - center)
        else:
            return -1.0  # Penalty for not detecting any chevron

        # Reward is inversely proportional to the deviation from the center
        reward = 1.0 / (1 + deviation)
        return reward

    # This function takes as input an image containing a chevron and returns the position of the red and blue parts of the chevron (if they exist) as a tuple. If the red or blue chevron is not detected, the corresponding position is None.
    def get_chevron_info(self, img):
        # Load the template images in color
        templates = [cv2.imread(f'src/data/chevron{i}.png', cv2.IMREAD_COLOR) for i in range(1, 4)]
        if any(template is None for template in templates):
            print("Error: One or more template images not found.")
            return None, None

        # Extract the default speed up color from the first template image
        speed_up_colour = templates[0][0, 0]

        # Perform template matching for each template image
        results = [cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) for template in templates]

        # Set a threshold for match quality
        threshold = 0.8

        # Find the coordinates of the best matches for each result
        locs = [np.where(result >= threshold) for result in results]

        # Combine the coordinates from all results
        loc = np.hstack(locs)

        # If no matches are found, return None
        if not loc[0].any() and not loc[1].any():
            return None, None

        # Get the x-coordinates of the detected chevrons
        position_red = loc[1][0] + templates[0].shape[1] // 2
        position_blue = loc[1][-1] + templates[0].shape[1] // 2

        # Get the colour of the detected chevrons
        colour_red = img[position_red, position_blue]
        colour_blue = img[position_blue, position_blue]

        # Check if the detected colour matches the default speed up color
        is_speed_up_colour = np.all(colour_red == speed_up_colour) or np.all(colour_blue == speed_up_colour)

        return position_red, position_blue, is_speed_up_colour


    def is_done(self, img):
        # TODO: Implement image processing techniques to detect if the car
        # goes off the road or crashes in the captured image
        done = False # Still need to implement this
        return done

    def draw_controller(self, img, action, position):
        center_x, center_y = position

        # Draw the controller base
        cv2.circle(img, (center_x, center_y), 50, (255, 255, 255), 2)

        # Draw the action indicator
        if action == 'accelerate':
            cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - 30), (0, 255, 0), 2)
        elif action == 'brake':
            cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y + 30), (0, 0, 255), 2)
        elif action == 'left':
            cv2.arrowedLine(img, (center_x, center_y), (center_x - 30, center_y), (255, 0, 0), 2)
        elif action == 'right':
            cv2.arrowedLine(img, (center_x, center_y), (center_x + 30, center_y), (255, 0, 0), 2)

    def display_overlays(self, img, predicted_speed_action, actual_speed_action, predicted_steering_action, actual_steering_action, position_red, position_blue):
            # Create a copy of the image to avoid modifying the original
            overlay_img = img.copy()

            # Draw the virtual controller for speed actions
            self.draw_controller(overlay_img, actual_speed_action, position=(100, overlay_img.shape[0] - 100))

            # Draw the virtual controller for steering actions
            self.draw_controller(overlay_img, actual_steering_action, position=(overlay_img.shape[1] - 100, overlay_img.shape[0] - 100))

            # Draw the chevron positions on the overlay image
            self.draw_chevron_positions(overlay_img, position_red, position_blue)

            # Display the overlay image in a separate window
            window_name = 'Overlays'
            cv2.imshow(window_name, overlay_img)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()


            

    def draw_chevron_positions(self, img, position_red, position_blue):
        # Draw the red chevron position
        if position_red is not None:
            cv2.circle(img, (position_red, self.monitor['height'] // 2), 10, (0, 0, 255), -1)

        # Draw the blue chevron position
        if position_blue is not None:
            cv2.circle(img, (position_blue, self.monitor['height'] // 2), 10, (255, 0, 0), -1)

        # If either of the chevrons are None, return False
        if position_red is None or position_blue is None:
            return False
        else:
            return True


# Update the display method to include the virtual controller
    def display(self, img, predicted_speed_action=None, actual_speed_action=None, predicted_steering_action=None, actual_steering_action=None, position_red=None, position_blue=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if predicted_speed_action is not None:
            cv2.putText(img, f"Predicted Speed Action: {predicted_speed_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if actual_speed_action is not None:
            cv2.putText(img, f"Actual Speed Action: {actual_speed_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if predicted_steering_action is not None:
            cv2.putText(img, f"Predicted Steering Action: {predicted_steering_action}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if actual_steering_action is not None:
            cv2.putText(img, f"Actual Steering Action: {actual_steering_action}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if position_red is not None:
            cv2.circle(img, (position_red, self.monitor['height']//2), 10, (0, 0, 255), 2)
        if position_blue is not None:
            cv2.circle(img, (position_blue, self.monitor['height']//2), 10, (255, 0, 0), 2)
        self.display_overlays(img, predicted_speed_action, actual_speed_action, predicted_steering_action, actual_steering_action, position_red, position_blue)

    def display_overlays(self, img, predicted_speed_action, actual_speed_action, predicted_steering_action, actual_steering_action, position_red, position_blue):
        cv2.imshow("img", img)
        cv2.waitKey(1)


