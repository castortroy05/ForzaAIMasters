"""
Unified Training Loop for Racing Agent

This module implements the main training loop with:
- Progressive difficulty adjustment
- Curriculum learning
- Model checkpointing
- Performance tracking and visualization
- Graceful error handling
"""

import time
import numpy as np
from collections import deque
from image_processing import preprocess_input_data
from unified_rewards import UnifiedRewardSystem, CurriculumRewardSystem


def train_racing_agent(
    agent,
    num_episodes=1000,
    save_frequency=50,
    verbose=True,
    use_curriculum=True,
    visualization=True
):
    """
    Train the unified racing agent from novice to pro level.

    Training stages:
    1. Novice (Episodes 0-200): Learn to stay on track
    2. Intermediate (Episodes 200-500): Learn speed control
    3. Advanced (Episodes 500-1000): Optimize racing lines
    4. Pro (Episodes 1000+): Master lap times

    Parameters:
    - agent (UnifiedRacingAgent): The agent to train
    - num_episodes (int): Total number of episodes
    - save_frequency (int): Save checkpoint every N episodes
    - verbose (bool): Print detailed training info
    - use_curriculum (bool): Use curriculum learning
    - visualization (bool): Display game overlays

    Returns:
    - training_history (dict): Complete training metrics
    """

    print("\n" + "="*80)
    print("UNIFIED RACING AGENT TRAINING".center(80))
    print("="*80 + "\n")

    # Initialize reward system
    if use_curriculum:
        reward_system = CurriculumRewardSystem(agent.game_env.game_window.monitor['width'])
        print("Using CURRICULUM LEARNING - Progressive difficulty")
    else:
        reward_system = UnifiedRewardSystem(
            agent.game_env.game_window.monitor['width'],
            difficulty='progressive'
        )
        print("Using PROGRESSIVE REWARDS")

    # Training history tracking
    training_history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_losses': [],
        'epsilon_values': [],
        'best_episode': 0,
        'best_reward': -float('inf')
    }

    # Moving average window for performance evaluation
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Checkpoint saves: Every {save_frequency} episodes")
    print(f"Target network updates: Every {agent.target_update_freq} steps\n")

    start_time = time.time()

    try:
        for episode in range(num_episodes):
            # Reset environment for new episode
            agent.game_env.reset_episode()
            reward_system.reset_episode()

            # Capture initial state
            screen = agent.game_env.capture()
            state = preprocess_input_data(screen)

            # Episode variables
            episode_reward = 0.0
            episode_length = 0
            episode_losses = []
            done = False

            episode_start_time = time.time()

            # ===== EPISODE LOOP =====
            while not done:
                # Choose action (epsilon-greedy)
                action_idx, (steering, speed) = agent.choose_action(state)

                # Execute action in game
                agent.simulate_action(steering, speed)

                # Small delay for game to update (adjust based on game speed)
                time.sleep(0.05)

                # Observe next state
                next_screen = agent.game_env.capture()
                next_state = preprocess_input_data(next_screen)

                # Get game info (chevron detection)
                try:
                    chevron_result = agent.game_env.get_chevron_info(next_screen)
                    if len(chevron_result) == 4:
                        position_red, position_blue, is_speed_up_colour, _ = chevron_result
                    else:
                        position_red, position_blue, is_speed_up_colour = chevron_result
                except Exception as e:
                    print(f"Warning: Chevron detection failed: {e}")
                    position_red, position_blue, is_speed_up_colour = None, None, False

                # Check if episode is done
                done = agent.game_env.is_done(next_screen)

                # Compute reward
                reward, reward_info = reward_system.compute_reward(
                    position_red,
                    position_blue,
                    is_speed_up_colour,
                    steering,
                    speed,
                    next_state=next_state,
                    done=done
                )

                # Store experience
                agent.store_transition(state, action_idx, reward, next_state, done)

                # Learn from experience
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1

                # Display visualization
                if visualization:
                    try:
                        agent.game_env.display_overlays(
                            next_screen,
                            speed_action=speed,
                            steering_action=steering,
                            position_red=position_red,
                            position_blue=position_blue
                        )
                    except Exception as e:
                        if verbose and episode == 0:
                            print(f"Visualization disabled: {e}")
                        visualization = False

                # Verbose step info (only for first few episodes)
                if verbose and episode < 3 and episode_length % 50 == 0:
                    print(f"  Step {episode_length}: "
                          f"Reward={reward:.2f}, "
                          f"Steering={steering:.2f}, "
                          f"Speed={speed:.2f}, "
                          f"Pos=({position_red}, {position_blue})")

            # ===== END EPISODE =====

            episode_duration = time.time() - episode_start_time

            # Decay epsilon after episode
            agent.decay_epsilon()

            # Update curriculum if using curriculum learning
            if use_curriculum and hasattr(reward_system, 'advance_curriculum'):
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                reward_system.advance_curriculum(episode, avg_reward)

            # Record metrics
            recent_rewards.append(episode_reward)
            recent_lengths.append(episode_length)
            training_history['episode_rewards'].append(episode_reward)
            training_history['episode_lengths'].append(episode_length)
            training_history['episode_losses'].append(np.mean(episode_losses) if episode_losses else 0)
            training_history['epsilon_values'].append(agent.eps)

            # Update agent metrics
            agent.episode_rewards.append(episode_reward)
            agent.episode_lengths.append(episode_length)
            agent.episode_count += 1

            # Check for best performance
            if episode_reward > training_history['best_reward']:
                training_history['best_reward'] = episode_reward
                training_history['best_episode'] = episode
                agent.best_reward = episode_reward

                # Save best model
                agent.save_model('best_model.h5')
                print(f"  *** NEW BEST MODEL (Reward: {episode_reward:.2f}) ***")

            # Print episode summary
            avg_recent_reward = np.mean(recent_rewards)
            avg_recent_length = np.mean(recent_lengths)

            print(f"\nEpisode {episode + 1}/{num_episodes} "
                  f"[{episode_duration:.1f}s]")
            print(f"  Reward: {episode_reward:.2f} "
                  f"(Avg: {avg_recent_reward:.2f}, "
                  f"Best: {training_history['best_reward']:.2f})")
            print(f"  Steps: {episode_length} "
                  f"(Avg: {avg_recent_length:.1f})")
            print(f"  Epsilon: {agent.eps:.4f}")
            print(f"  Memory size: {len(agent.memory)}/{agent.mem_size}")
            print(f"  Learn steps: {agent.learn_step}")

            if episode_losses:
                print(f"  Avg loss: {np.mean(episode_losses):.4f}")

            # Save checkpoint
            if (episode + 1) % save_frequency == 0:
                agent.save_training_state(episode + 1)
                print(f"\n  Checkpoint saved at episode {episode + 1}")

                # Print progress summary
                elapsed_time = time.time() - start_time
                episodes_per_hour = (episode + 1) / (elapsed_time / 3600)
                print(f"\n  Progress: {(episode + 1) / num_episodes * 100:.1f}%")
                print(f"  Elapsed time: {elapsed_time / 60:.1f} minutes")
                print(f"  Episodes/hour: {episodes_per_hour:.1f}")
                print(f"  Estimated time remaining: "
                      f"{(num_episodes - episode - 1) / episodes_per_hour * 60:.1f} minutes")

            # Periodic detailed stats
            if (episode + 1) % 100 == 0:
                print_training_summary(training_history, episode + 1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current progress...")
        agent.save_training_state(episode)

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving emergency checkpoint...")
        try:
            agent.save_training_state(episode)
        except:
            print("Failed to save emergency checkpoint")

    finally:
        # Final save
        print("\n" + "="*80)
        print("TRAINING COMPLETE".center(80))
        print("="*80)

        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time / 60:.1f} minutes")
        print(f"Total episodes: {len(training_history['episode_rewards'])}")
        print(f"Best reward: {training_history['best_reward']:.2f} "
              f"(Episode {training_history['best_episode']})")
        print(f"Final epsilon: {agent.eps:.4f}")

        # Final save
        agent.save_training_state(agent.episode_count)

        return training_history


def print_training_summary(history, current_episode):
    """
    Print detailed training summary.

    Parameters:
    - history (dict): Training history
    - current_episode (int): Current episode number
    """
    print("\n" + "-"*80)
    print(f"TRAINING SUMMARY - Episode {current_episode}".center(80))
    print("-"*80)

    rewards = history['episode_rewards']
    lengths = history['episode_lengths']

    if len(rewards) > 0:
        # Recent performance (last 100 episodes)
        recent_start = max(0, len(rewards) - 100)
        recent_rewards = rewards[recent_start:]
        recent_lengths = lengths[recent_start:]

        print(f"\nRecent Performance (Last {len(recent_rewards)} episodes):")
        print(f"  Average reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
        print(f"  Max reward: {np.max(recent_rewards):.2f}")
        print(f"  Min reward: {np.min(recent_rewards):.2f}")
        print(f"  Average length: {np.mean(recent_lengths):.1f} steps")

        # Overall performance
        print(f"\nOverall Performance:")
        print(f"  Average reward: {np.mean(rewards):.2f}")
        print(f"  Best reward: {history['best_reward']:.2f} (Episode {history['best_episode']})")
        print(f"  Average length: {np.mean(lengths):.1f} steps")

        # Improvement trend
        if len(rewards) >= 200:
            first_100_avg = np.mean(rewards[:100])
            last_100_avg = np.mean(rewards[-100:])
            improvement = ((last_100_avg - first_100_avg) / abs(first_100_avg)) * 100

            print(f"\nLearning Progress:")
            print(f"  First 100 episodes avg: {first_100_avg:.2f}")
            print(f"  Last 100 episodes avg: {last_100_avg:.2f}")
            print(f"  Improvement: {improvement:+.1f}%")

    print("-"*80 + "\n")


def evaluate_agent(agent, num_episodes=10, render=True):
    """
    Evaluate trained agent performance without exploration.

    Parameters:
    - agent (UnifiedRacingAgent): Trained agent
    - num_episodes (int): Number of evaluation episodes
    - render (bool): Display game overlays

    Returns:
    - eval_results (dict): Evaluation metrics
    """
    print("\n" + "="*80)
    print("AGENT EVALUATION".center(80))
    print("="*80 + "\n")

    # Disable exploration
    original_eps = agent.eps
    agent.eps = 0.0  # Pure exploitation

    eval_rewards = []
    eval_lengths = []

    for episode in range(num_episodes):
        agent.game_env.reset_episode()
        screen = agent.game_env.capture()
        state = preprocess_input_data(screen)

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Choose best action (no exploration)
            action_idx, (steering, speed) = agent.choose_action(state)
            agent.simulate_action(steering, speed)

            time.sleep(0.05)

            next_screen = agent.game_env.capture()
            next_state = preprocess_input_data(next_screen)

            # Minimal reward calculation for evaluation
            try:
                position_red, position_blue, is_speed_up_colour = agent.game_env.get_chevron_info(next_screen)[:3]
                # Simple progress reward
                if position_red is not None or position_blue is not None:
                    episode_reward += 1.0
            except:
                pass

            done = agent.game_env.is_done(next_screen)
            state = next_state
            episode_length += 1

            if render:
                try:
                    agent.game_env.display_overlays(
                        next_screen,
                        speed_action=speed,
                        steering_action=steering,
                        position_red=position_red if 'position_red' in locals() else None,
                        position_blue=position_blue if 'position_blue' in locals() else None
                    )
                except:
                    pass

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

        print(f"Evaluation Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, Steps={episode_length}")

    # Restore original epsilon
    agent.eps = original_eps

    eval_results = {
        'avg_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'min_reward': np.min(eval_rewards),
        'avg_length': np.mean(eval_lengths)
    }

    print("\n" + "-"*80)
    print("EVALUATION RESULTS".center(80))
    print("-"*80)
    print(f"Average reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Max reward: {eval_results['max_reward']:.2f}")
    print(f"Min reward: {eval_results['min_reward']:.2f}")
    print(f"Average length: {eval_results['avg_length']:.1f} steps")
    print("-"*80 + "\n")

    return eval_results
