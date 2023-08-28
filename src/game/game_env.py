from image_capture import ImageCapture
from detection import Detection
from overlay import Overlay
from rewards import RewardSystem
from game_window import GameWindow  # Ensure the GameWindow class is imported

class GameEnvironment:
    def __init__(self):
        self.game_window = GameWindow()  # Get the game window's monitor details
        self.img_capture = ImageCapture(self.game_window.monitor)  # Pass the monitor details to ImageCapture
        self.detection = Detection()
        self.overlay = Overlay(self.img_capture.monitor, Detection.templates)
        self.rewards = RewardSystem(self.game_window.monitor['width'])  # Pass the monitor width to RewardSystem

    def capture(self):
        return self.img_capture.capture()

    def get_chevron_info(self, img):
        return self.detection.get_chevron_info(img)

    def display_overlays(self, img, speed_action=None, steering_action=None, position_red=None, position_blue=None):
        self.overlay.display_overlays(img, speed_action=speed_action, steering_action=steering_action, position_red=position_red, position_blue=position_blue)

    def compute_speed_reward(self, *args):
        return self.rewards.speed_reward(*args)

    def compute_steering_reward(self, *args):
        return self.rewards.steering_reward(*args)
    
    def is_done(self, img):
        """Check if the game is over (e.g., car crash). Placeholder for now."""
        return False

    
    @property
    def expected_shape(self):
        return self.img_capture.expected_shape
