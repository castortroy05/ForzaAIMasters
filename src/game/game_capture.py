from mss import mss
import cv2
import numpy as np

class GameCapture:
    def __init__(self, top, left, width, height):
        self.sct = mss()
        self.monitor = {"top": top, "left": left, "width": width, "height": height}

    def capture(self):
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        return img

    def display(self, img):
        cv2.imshow('Game Capture', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
