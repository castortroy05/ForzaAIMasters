import pygetwindow as gw

class GameWindow:
    def __init__(self, window_title="Forza Motorsport 7"):
        self.window_title = window_title
        self.monitor = self.initialize_game_window()

    def initialize_game_window(self):
        game_window = gw.getWindowsWithTitle(self.window_title)
        if game_window:
            game_window = game_window[0]
            return {
                "top": game_window.top,
                "left": game_window.left,
                "width": game_window.width,
                "height": game_window.height
            }
        else:
            raise ValueError(f"Could not find game window with title: {self.window_title}")
