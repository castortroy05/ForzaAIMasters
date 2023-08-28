class RewardSystem:
    def __init__(self, monitor_width):
        self.center = monitor_width // 2

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
        if position_red is not None:
            deviation = abs(position_red - self.center)
        elif position_blue is not None:
            deviation = abs(position_blue - self.center)
        else:
            return -5.0  # Increased penalty for not detecting any chevron

        # Reward is inversely proportional to the deviation from the center
        # We'll square the inverse to emphasize the importance of being close to the center
        reward = 1.0 / (1 + deviation**2)
        return reward
