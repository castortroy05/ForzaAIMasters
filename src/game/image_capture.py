from mss import mss
import numpy as np

class ImageCapture:
    def __init__(self, monitor):
        self.sct = mss()
        self.monitor = monitor
        self.expected_shape = (self.monitor['height'], self.monitor['width'], 3)

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