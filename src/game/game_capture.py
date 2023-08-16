from mss import mss
import cv2
import numpy as np
import pygetwindow as gw

class GameEnvironment:
    def __init__(self):
        self.sct = mss()
        window_title = "Forza Motorsport 7"
        game_window = gw.getWindowsWithTitle(window_title)[0]
        self.monitor = {
            "top": game_window.top,
            "left": game_window.left,
            "width": game_window.width,
            "height": game_window.height
        }

    def capture(self):
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        return img
    
    def speed_reward(self, position_red, position_blue, action):
        if position_red is not None and action == 'brake':
            return 1.0
        elif position_blue is not None and action == 'accelerate':
            return 1.0
        else:
            return -1.0  # Penalty for incorrect action

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


    def get_chevron_info(self, img):
        # Define the color ranges for the chevron (you may need to adjust these values)
        lower_red = np.array([0, 0, 200])
        upper_red = np.array([50, 50, 255])
        lower_green = np.array([0, 200, 0])
        upper_green = np.array([50, 255, 50])

        # Create masks for the red and green colors
        mask_red = cv2.inRange(img, lower_red, upper_red)
        mask_green = cv2.inRange(img, lower_green, upper_green)

        # Use contour detection to identify the chevron
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are detected, return None
        if not contours_red and not contours_green:
            return None

        # Find the largest contour for each color (assuming it corresponds to the chevron)
        if contours_red:
            largest_contour_red = max(contours_red, key=cv2.contourArea)
            x_red, y_red, w_red, h_red = cv2.boundingRect(largest_contour_red)
            position_red = x_red + w_red // 2
        else:
            position_red = None

        if contours_green:
            largest_contour_green = max(contours_green, key=cv2.contourArea)
            x_green, y_green, w_green, h_green = cv2.boundingRect(largest_contour_green)
            position_green = x_green + w_green // 2
        else:
            position_green = None

        return position_red, position_green


    

    def is_done(self, img):
        # TODO: Implement image processing techniques to detect if the car
        # goes off the road or crashes in the captured image
        done = False # You need to implement this
        return done

    def display(self, img):
        cv2.imshow('Game Capture', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
