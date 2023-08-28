import cv2
import numpy as np
from detection import Detection
class Overlay:
    
    def __init__(self, monitor, templates):
        self.monitor = monitor
        self.templates = templates

    def display_overlays(self, img, speed_action=None, steering_action=None, position_red=None, position_blue=None, detected_chevrons=None):
        """Display overlays on the image."""
        overlay_img = img.copy()
        templates = Detection.templates

        # Display speed action
        if speed_action:
            cv2.putText(overlay_img, f"Speed Action: {speed_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display steering action
        if steering_action:
            cv2.putText(overlay_img, f"Steering Action: {steering_action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Highlight detected red position
        if position_red is not None:
            cv2.circle(overlay_img, (position_red, self.monitor['height']//2), 10, (0, 0, 255), 2)

        # Highlight detected blue position
        if position_blue is not None:
            cv2.circle(overlay_img, (position_blue, self.monitor['height']//2), 10, (255, 0, 0), 2)

            
        # Draw rectangles around detected chevrons
        if detected_chevrons:
            for pt in detected_chevrons:
                cv2.rectangle(overlay_img, pt, (pt[0] + templates[0].shape[1], pt[1] + templates[0].shape[0]), (0, 255, 0), 2)


        # Display the overlay image
        cv2.imshow("Overlays", overlay_img)
        cv2.waitKey(1)
