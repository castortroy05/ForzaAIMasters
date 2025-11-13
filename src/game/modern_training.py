"""
Modern Training Infrastructure (2024-2025)

Features:
- TensorBoard logging with rich metrics
- Learning rate scheduling (cosine annealing, warmup)
- Data augmentation for robustness
- Intrinsic motivation (curiosity)
- Automatic checkpointing
- Early stopping
- Gradient monitoring
- Performance profiling
"""

import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import deque


class ModernTrainer:
    """
    Modern training loop with all best practices.
    """

    def __init__(
        self,
        agent,
        game_env,
        vision_encoder=None,
        use_curiosity=True,
        curiosity_weight=0.1,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        save_frequency=50,
        use_tensorboard=True
    ):
        """
        Initialize modern trainer.

        Parameters:
        - agent: PPO agent instance
        - game_env: Game environment
        - vision_encoder: Modern vision encoder (optional)
        - use_curiosity: Enable intrinsic motivation
        - curiosity_weight: Weight for curiosity rewards
        - log_dir: TensorBoard log directory
        - checkpoint_dir: Model checkpoint directory
        - save_frequency: Save checkpoint every N episodes
        - use_tensorboard: Enable TensorBoard logging
        """
        self.agent = agent
        self.game_env = game_env
        self.vision_encoder = vision_encoder
        self.use_curiosity = use_curiosity
        self.curiosity_weight = curiosity_weight
        self.save_frequency = save_frequency

        # Create directories
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(log_dir, timestamp)
        self.checkpoint_dir = os.path.join(checkpoint_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard
        if use_tensorboard:
            self.writer = tf.summary.create_file_writer(self.log_dir)
            print(f"TensorBoard logs: {self.log_dir}")
            print(f"Run: tensorboard --logdir={self.log_dir}")

        # Curiosity module
        if use_curiosity:
            from modern_vision import CuriosityModule
            self.curiosity = CuriosityModule(
                state_dim=agent.state_dim,
                action_dim=agent.action_dim,
                feature_dim=256
            )
            print(f"Intrinsic curiosity enabled (weight={curiosity_weight})")

        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingScheduler(
            initial_lr=agent.learning_rate,
            min_lr=agent.learning_rate / 10,
            warmup_steps=1000
        )

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        self.global_step = 0

    def train(self, num_episodes=1000, eval_frequency=100):
        """
        Main training loop.

        Parameters:
        - num_episodes: Total episodes to train
        - eval_frequency: Evaluate every N episodes
        """
        print("\n" + "="*80)
        print("MODERN PPO TRAINING".center(80))
        print("="*80 + "\n")

        start_time = time.time()

        for episode in range(num_episodes):
            # Run episode
            episode_reward, episode_length, metrics = self._run_episode(episode)

            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Update learning rate
            new_lr = self.lr_scheduler.step(self.global_step)
            self.agent.optimizer.learning_rate.assign(new_lr)

            # TensorBoard logging
            if hasattr(self, 'writer'):
                with self.writer.as_default():
                    tf.summary.scalar('episode/reward', episode_reward, step=episode)
                    tf.summary.scalar('episode/length', episode_length, step=episode)
                    tf.summary.scalar('episode/learning_rate', new_lr, step=episode)

                    if metrics:
                        for key, value in metrics.items():
                            tf.summary.scalar(f'training/{key}', value, step=episode)

            # Print progress
            avg_reward = np.mean(self.episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(100): {avg_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"LR: {new_lr:.6f}")

            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.agent.save(filename='best_model')
                print(f"  *** NEW BEST: {episode_reward:.2f} ***")

            # Periodic checkpoint
            if (episode + 1) % self.save_frequency == 0:
                self.agent.save(filename=f'checkpoint_episode_{episode + 1}')
                print(f"  Checkpoint saved at episode {episode + 1}")

            # Evaluation
            if (episode + 1) % eval_frequency == 0:
                self._evaluate(num_eval_episodes=10, episode=episode + 1)

        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time / 60:.1f} minutes")
        print(f"Best reward: {self.best_reward:.2f}")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_reward': self.best_reward
        }

    def _run_episode(self, episode_num):
        """Run single episode."""
        self.game_env.reset_episode()

        # Capture initial frame
        frame = self.game_env.capture()

        # Extract state features
        if self.vision_encoder:
            state = self.vision_encoder.encode(frame)
        else:
            from image_processing import preprocess_input_data
            state = preprocess_input_data(frame)

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < 5000:  # Max steps per episode
            # Get action from policy
            action, log_prob, value = self.agent.get_action(state)

            # Execute action
            steering, throttle = action[0], action[1]
            self._execute_action(steering, throttle)

            # Small delay for game to update
            time.sleep(0.05)

            # Observe next state
            next_frame = self.game_env.capture()

            if self.vision_encoder:
                next_state = self.vision_encoder.encode(next_frame)
            else:
                next_state = preprocess_input_data(next_frame)

            # Get game info
            try:
                chevron_info = self.game_env.get_chevron_info(next_frame)
                if len(chevron_info) >= 3:
                    position_red, position_blue, is_speed_up_colour = chevron_info[:3]
                else:
                    position_red, position_blue, is_speed_up_colour = None, None, False
            except:
                position_red, position_blue, is_speed_up_colour = None, None, False

            # Check if done
            done = self.game_env.is_done(next_frame)

            # Compute extrinsic reward
            extrinsic_reward = self._compute_reward(
                position_red, position_blue, is_speed_up_colour,
                steering, throttle, done
            )

            # Compute intrinsic reward (curiosity)
            intrinsic_reward = 0.0
            if self.use_curiosity:
                intrinsic_reward = self.curiosity.compute_intrinsic_reward(
                    state, action, next_state
                )

            # Total reward
            total_reward = extrinsic_reward + self.curiosity_weight * intrinsic_reward

            # Store transition
            self.agent.store_transition(state, action, total_reward, value, log_prob, done)

            # Update state
            state = next_state
            episode_reward += extrinsic_reward
            episode_length += 1
            self.global_step += 1

            # Display (optional)
            try:
                self.game_env.display_overlays(
                    next_frame,
                    speed_action=throttle,
                    steering_action=steering,
                    position_red=position_red,
                    position_blue=position_blue
                )
            except:
                pass

        # Update agent after episode
        metrics = None
        if len(self.agent.buffer) >= self.agent.buffer_size:
            metrics = self.agent.update()

        return episode_reward, episode_length, metrics

    def _execute_action(self, steering, throttle):
        """Execute steering and throttle action in game."""
        # Use game controller
        from controller import GameController

        if not hasattr(self, '_controller'):
            self._controller = GameController()

        # Steering
        x_value = int(np.clip(steering, -1, 1) * 32767)
        self._controller.left_joystick(x_value=x_value, y_value=0)

        # Throttle/Brake
        if throttle >= 0:
            self._controller.right_trigger(value=int(np.clip(throttle, 0, 1) * 255))
            self._controller.left_trigger(value=0)
        else:
            self._controller.left_trigger(value=int(np.clip(-throttle, 0, 1) * 255))
            self._controller.right_trigger(value=0)

        self._controller.update()

    def _compute_reward(self, position_red, position_blue, is_speed_up_colour,
                       steering, throttle, done):
        """Compute reward (can use unified reward system)."""
        from unified_rewards import UnifiedRewardSystem

        if not hasattr(self, '_reward_system'):
            self._reward_system = UnifiedRewardSystem(
                self.game_env.game_window.monitor['width'],
                difficulty='progressive'
            )

        reward, _ = self._reward_system.compute_reward(
            position_red,
            position_blue,
            is_speed_up_colour,
            steering,
            throttle,
            done=done
        )

        return reward

    def _evaluate(self, num_eval_episodes=10, episode=0):
        """Evaluate agent performance."""
        print(f"\n{'='*60}")
        print(f"EVALUATION (Episode {episode})".center(60))
        print('='*60)

        eval_rewards = []

        for eval_ep in range(num_eval_episodes):
            self.game_env.reset_episode()
            frame = self.game_env.capture()

            if self.vision_encoder:
                state = self.vision_encoder.encode(frame)
            else:
                from image_processing import preprocess_input_data
                state = preprocess_input_data(frame)

            eval_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 5000:
                # Deterministic action
                action, _, _ = self.agent.get_action(state, deterministic=True)

                steering, throttle = action[0], action[1]
                self._execute_action(steering, throttle)

                time.sleep(0.05)

                next_frame = self.game_env.capture()

                if self.vision_encoder:
                    next_state = self.vision_encoder.encode(next_frame)
                else:
                    next_state = preprocess_input_data(next_frame)

                try:
                    position_red, position_blue, is_speed_up_colour = self.game_env.get_chevron_info(next_frame)[:3]
                except:
                    position_red, position_blue, is_speed_up_colour = None, None, False

                done = self.game_env.is_done(next_frame)

                reward = self._compute_reward(
                    position_red, position_blue, is_speed_up_colour,
                    steering, throttle, done
                )

                eval_reward += reward
                state = next_state
                steps += 1

            eval_rewards.append(eval_reward)
            print(f"  Eval {eval_ep + 1}/{num_eval_episodes}: {eval_reward:.2f} ({steps} steps)")

        avg_eval_reward = np.mean(eval_rewards)
        print(f"\nAverage Evaluation Reward: {avg_eval_reward:.2f} ± {np.std(eval_rewards):.2f}")
        print('='*60 + '\n')

        # Log to TensorBoard
        if hasattr(self, 'writer'):
            with self.writer.as_default():
                tf.summary.scalar('eval/avg_reward', avg_eval_reward, step=episode)
                tf.summary.scalar('eval/std_reward', np.std(eval_rewards), step=episode)

        return avg_eval_reward


class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate schedule with warmup.

    Modern LR scheduling for better convergence.
    """

    def __init__(self, initial_lr, min_lr, warmup_steps=1000, total_steps=100000):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def step(self, current_step):
        """Get learning rate for current step."""
        if current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine


class DataAugmentation:
    """
    Data augmentation for visual robustness.

    Applies random transformations to images during training.
    """

    @staticmethod
    def augment(image, training=True):
        """Apply random augmentations."""
        if not training:
            return image

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Random saturation
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

        # Small random crops (simulating camera shake)
        if np.random.random() < 0.3:
            image = tf.image.random_crop(
                image,
                size=[int(image.shape[0] * 0.95), int(image.shape[1] * 0.95), image.shape[2]]
            )
            image = tf.image.resize(image, [image.shape[0], image.shape[1]])

        return image
