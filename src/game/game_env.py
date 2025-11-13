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
        """
        Check if episode should terminate (crash, off-track, or max steps).

        Detection methods:
        1. Check for crash screen indicators
        2. Check if no chevrons detected for extended period (off-track)
        3. Maximum step limit
        """
        # Try to detect chevrons
        position_red, position_blue, is_speed_up_colour = self.get_chevron_info(img)

        # If no chevrons detected, might be crashed or off-track
        if position_red is None and position_blue is None:
            # Increment no-chevron counter
            if not hasattr(self, '_no_chevron_count'):
                self._no_chevron_count = 0
            self._no_chevron_count += 1

            # If no chevrons for 30 consecutive frames, consider crashed/off-track
            if self._no_chevron_count > 30:
                print("Episode terminated: No chevrons detected (likely crashed or off-track)")
                return True
        else:
            # Reset counter if chevrons are detected
            self._no_chevron_count = 0

        # Check for maximum step limit
        if not hasattr(self, '_episode_steps'):
            self._episode_steps = 0
        self._episode_steps += 1

        # Maximum episode length (prevent infinite loops)
        MAX_STEPS = 5000
        if self._episode_steps >= MAX_STEPS:
            print(f"Episode terminated: Reached maximum steps ({MAX_STEPS})")
            return True

        # TODO: Add crash screen detection using template matching or color analysis
        # E.g., check for "REWIND" button, specific UI elements, etc.

        return False

    def reset_episode(self):
        """Reset episode-specific counters."""
        self._no_chevron_count = 0
        self._episode_steps = 0

    
    @property
    def expected_shape(self):
        return self.img_capture.expected_shape
