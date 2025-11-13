"""
Unified Reward System for Racing Agent

This reward function considers both steering and speed actions together,
encouraging coordinated racing behavior. It implements progressive reward
shaping to guide learning from novice to pro level.

Reward Components:
1. Track positioning (stay centered on track)
2. Speed optimization (go fast on straights, slow on turns)
3. Smoothness (discourage jerky control)
4. Progress (forward movement)
5. Crash penalty (negative terminal reward)
"""

import numpy as np


class UnifiedRewardSystem:
    """
    Unified reward system that evaluates coordinated steering + speed actions.

    This reward function is designed to teach the agent progressively:
    - Novice: Learn to stay on track
    - Intermediate: Learn to balance speed and steering
    - Advanced: Learn to optimize racing lines
    - Pro: Learn to maximize lap times
    """

    def __init__(self, monitor_width, difficulty='progressive'):
        """
        Initialize reward system.

        Parameters:
        - monitor_width (int): Width of game screen for center calculation
        - difficulty (str): Reward shaping difficulty
                           'novice' - Heavy guidance, focus on staying on track
                           'intermediate' - Balance speed and control
                           'advanced' - Optimize racing lines
                           'progressive' - Automatically adjust based on performance
        """
        self.center = monitor_width // 2
        self.difficulty = difficulty
        self.prev_speed = 0.0
        self.prev_steering = 0.0
        self.step_count = 0
        self.total_reward = 0.0

        # Reward weights for different difficulty levels
        self.reward_weights = {
            'novice': {
                'position': 2.0,      # Heavy weight on staying centered
                'speed': 0.5,         # Light weight on speed
                'smoothness': 0.3,    # Some smoothness encouraged
                'progress': 0.2,      # Minimal progress reward
                'crash': -50.0        # Moderate crash penalty
            },
            'intermediate': {
                'position': 1.5,
                'speed': 1.0,
                'smoothness': 0.5,
                'progress': 0.5,
                'crash': -100.0
            },
            'advanced': {
                'position': 1.0,
                'speed': 1.5,
                'smoothness': 0.7,
                'progress': 1.0,
                'crash': -200.0
            },
            'progressive': {
                'position': 2.0,
                'speed': 0.5,
                'smoothness': 0.3,
                'progress': 0.2,
                'crash': -50.0
            }
        }

        self.weights = self.reward_weights[difficulty]

    def compute_reward(
        self,
        position_red,
        position_blue,
        is_speed_up_colour,
        steering_action,
        speed_action,
        next_state=None,
        done=False
    ):
        """
        Compute unified reward considering both steering and speed.

        Parameters:
        - position_red (int): X-position of red chevron (if detected)
        - position_blue (int): X-position of blue chevron (if detected)
        - is_speed_up_colour (bool): Whether acceleration is recommended
        - steering_action (float): Steering action taken [-1, 1]
        - speed_action (float): Speed action taken [-1, 1]
        - next_state (array): Next state observation (optional)
        - done (bool): Whether episode terminated (crash/completion)

        Returns:
        - reward (float): Total reward for this step
        - reward_info (dict): Breakdown of reward components
        """

        # Initialize reward components
        position_reward = 0.0
        speed_reward = 0.0
        smoothness_reward = 0.0
        progress_reward = 0.0
        crash_penalty = 0.0

        # === 1. POSITION REWARD ===
        # Reward staying centered on track (using chevron positions)
        if position_red is not None:
            deviation = abs(position_red - self.center)
        elif position_blue is not None:
            deviation = abs(position_blue - self.center)
        else:
            # No chevrons detected - likely off track or poor detection
            deviation = self.center  # Maximum deviation
            position_reward = -2.0  # Penalty for poor visibility

        if position_red is not None or position_blue is not None:
            # Inverse squared deviation - heavily rewards staying centered
            # Max reward when centered, drops off quickly when off-center
            position_reward = 1.0 / (1.0 + (deviation / 100.0) ** 2)

        # === 2. SPEED REWARD ===
        # Encourage appropriate speed based on track conditions
        if is_speed_up_colour:
            # Straight section - reward acceleration
            if speed_action > 0.3:  # Accelerating
                speed_reward = speed_action * 1.0
            elif speed_action < 0:  # Braking when should accelerate
                speed_reward = speed_action * 0.5  # Penalty
            else:  # Coasting
                speed_reward = 0.0
        else:
            # Turn section - reward controlled speed
            if deviation < 50:  # If well-positioned
                # Can maintain some speed
                if 0.0 <= speed_action <= 0.5:
                    speed_reward = 0.5
                elif speed_action > 0.5:
                    speed_reward = -0.3  # Penalty for too fast in turn
            else:  # Poorly positioned
                # Should slow down
                if speed_action < 0.2:  # Braking or slow
                    speed_reward = 0.5
                else:  # Too fast while off-center
                    speed_reward = -0.5

        # === 3. SMOOTHNESS REWARD ===
        # Penalize jerky, erratic control
        steering_change = abs(steering_action - self.prev_steering)
        speed_change = abs(speed_action - self.prev_speed)

        # Small changes are good, large changes are penalized
        steering_smoothness = -steering_change * 0.5
        speed_smoothness = -speed_change * 0.3

        smoothness_reward = steering_smoothness + speed_smoothness

        # === 4. PROGRESS REWARD ===
        # Small constant reward for each step (encourages forward progress)
        progress_reward = 0.1

        # Bonus for going fast while centered
        if position_red is not None or position_blue is not None:
            if deviation < 100 and speed_action > 0.5:
                progress_reward += 0.5  # Bonus for fast + centered

        # === 5. CRASH PENALTY ===
        if done:
            crash_penalty = self.weights['crash']

        # === COMBINE REWARDS ===
        total_reward = (
            self.weights['position'] * position_reward +
            self.weights['speed'] * speed_reward +
            self.weights['smoothness'] * smoothness_reward +
            self.weights['progress'] * progress_reward +
            crash_penalty
        )

        # Update state for next step
        self.prev_steering = steering_action
        self.prev_speed = speed_action
        self.step_count += 1
        self.total_reward += total_reward

        # Reward breakdown for debugging/monitoring
        reward_info = {
            'total': total_reward,
            'position': position_reward * self.weights['position'],
            'speed': speed_reward * self.weights['speed'],
            'smoothness': smoothness_reward * self.weights['smoothness'],
            'progress': progress_reward * self.weights['progress'],
            'crash': crash_penalty,
            'deviation': deviation if (position_red is not None or position_blue is not None) else None,
            'steering_action': steering_action,
            'speed_action': speed_action
        }

        return total_reward, reward_info

    def update_difficulty(self, avg_reward, avg_episode_length):
        """
        Progressively adjust difficulty based on agent performance.

        Parameters:
        - avg_reward (float): Average reward over recent episodes
        - avg_episode_length (int): Average episode length
        """
        if self.difficulty != 'progressive':
            return  # Only auto-adjust in progressive mode

        # Criteria for difficulty progression
        if avg_reward > 50 and avg_episode_length > 500:
            # Agent is doing well - transition to intermediate
            self.weights = self.reward_weights['intermediate']
            print("Difficulty increased to INTERMEDIATE")
            self.difficulty = 'intermediate_locked'

        elif avg_reward > 150 and avg_episode_length > 1000:
            # Agent is excelling - transition to advanced
            self.weights = self.reward_weights['advanced']
            print("Difficulty increased to ADVANCED")
            self.difficulty = 'advanced_locked'

    def reset_episode(self):
        """Reset episode-specific state."""
        self.prev_speed = 0.0
        self.prev_steering = 0.0
        self.step_count = 0
        self.total_reward = 0.0

    def get_episode_stats(self):
        """Get episode statistics."""
        return {
            'total_reward': self.total_reward,
            'avg_reward_per_step': self.total_reward / max(1, self.step_count),
            'steps': self.step_count
        }


class CurriculumRewardSystem(UnifiedRewardSystem):
    """
    Advanced reward system with curriculum learning.

    Implements staged learning where tasks become progressively harder:
    Stage 1: Learn to keep car on track (position only)
    Stage 2: Learn to maintain speed (position + speed)
    Stage 3: Learn smooth control (position + speed + smoothness)
    Stage 4: Learn to optimize (all components + racing line)
    """

    def __init__(self, monitor_width):
        super().__init__(monitor_width, difficulty='progressive')
        self.curriculum_stage = 1
        self.stage_thresholds = {
            1: {'min_reward': 30, 'min_episodes': 50},
            2: {'min_reward': 80, 'min_episodes': 100},
            3: {'min_reward': 150, 'min_episodes': 150}
        }

    def advance_curriculum(self, episode, avg_reward):
        """
        Advance curriculum stage based on performance.

        Parameters:
        - episode (int): Current episode number
        - avg_reward (float): Average reward over recent episodes
        """
        if self.curriculum_stage >= 4:
            return  # Already at final stage

        threshold = self.stage_thresholds.get(self.curriculum_stage)
        if threshold and episode >= threshold['min_episodes'] and avg_reward >= threshold['min_reward']:
            self.curriculum_stage += 1
            print(f"\n{'='*60}")
            print(f"CURRICULUM ADVANCED TO STAGE {self.curriculum_stage}")
            print(f"{'='*60}\n")

            # Adjust reward weights for new stage
            if self.curriculum_stage == 2:
                self.weights['speed'] = 1.0
            elif self.curriculum_stage == 3:
                self.weights['smoothness'] = 0.7
            elif self.curriculum_stage == 4:
                self.weights['progress'] = 1.5
                self.weights['speed'] = 2.0
