import vgamepad as vg

class GameController:
    def __init__(self):
        self.controller = vg.VX360Gamepad()

    def simulate_steering_action(self, action):
        x_value = int(action * 32767)
        self.controller.left_joystick(x_value=x_value, y_value=0)
        self.controller.right_trigger(value=20)
        self.controller.update()

    def simulate_speed_action(self, action):
        if action >= 0:  # Accelerate
            self.controller.right_trigger(value=int(action * 255))
            self.controller.left_trigger(value=0)
        else:  # Brake
            self.controller.left_trigger(value=int(-action * 255))
            self.controller.right_trigger(value=0)
        self.controller.update()
