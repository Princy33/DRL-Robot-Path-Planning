import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch

# Allow imports from backend folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPISODES = 500
GRID_SIZE = 20
MAX_STEPS = 150

BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10

LEARNING_RATE = 0.001
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

RANDOM_SEED = 2026

MODEL_PATH = "models/dqn_random_curriculum_model.pth"
BEST_MODEL_PATH = "models/dqn_random_curriculum_best_model.pth"

CSV_PATH = "results/csv/dqn_random_curriculum_training_results.csv"


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results/csv", exist_ok=True)


# ============================================================
# CURRICULUM
# ============================================================

def get_obstacle_density(episode):
    """
    Gradually increase environment difficulty.
    """

    if episode <= 150:
        return 0.05

    elif episode <= 300:
        return 0.08

    elif episode <= 400:
        return 0.12

    else:
        return 0.15


# ============================================================
# ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS,
    random_obstacles=True,
    obstacle_density=0.05
)


# ============================================================
# AGENT
# ============================================================

agent = DQNAgent(
    state_size=405,
    action_size=4,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    replay_capacity=100000,
    batch_size=BATCH_SIZE,
    target_update_frequency=TARGET_UPDATE_FREQUENCY
)


# ============================================================
# TRAINING STORAGE
# ============================================================

results = []

best_moving_reward = -float("inf")
best_success_rate = 0.0

reward_history = []
success_history = []
steps_history = []

start_time = time.time()


# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 70)
print("CURRICULUM-BASED RANDOM DQN TRAINING")
print("=" * 70)

print(f"Grid Size              : {GRID_SIZE} x {GRID_SIZE}")
print(f"Maximum Steps          : {MAX_STEPS}")
print(f"Training Episodes      : {NUM_EPISODES}")
print(f"Batch Size             : {BATCH_SIZE}")
print(f"Learning Rate          : {LEARNING_RATE}")
print(f"Gamma                  : {GAMMA}")
print(f"Initial Epsilon        : {EPSILON_START}")
print(f"Minimum Epsilon        : {EPSILON_MIN}")
print(f"Epsilon Decay          : {EPSILON_DECAY}")
print(f"Random Seed            : {RANDOM_SEED}")
print("=" * 70)


for episode in range(1, NUM_EPISODES + 1):

    episode_start = time.time()

    # --------------------------------------------------------
    # Curriculum obstacle density
    # --------------------------------------------------------

    density = get_obstacle_density(episode)

    env.obstacle_density = density

    # Different seed for every episode
    episode_seed = RANDOM_SEED + episode

    state, info = env.reset(seed=episode_seed)

    total_reward = 0.0
    losses = []

    success = False
    collision = False

    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    for step in range(1, MAX_STEPS + 1):

        action = agent.select_action(
            state,
            training=True
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Store experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Train network
        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        state = next_state

        total_reward += reward

        # Goal reached
        if terminated and info.get("success", False):
            success = True

        # Collision
        if info.get("collision", False):
            collision = True

        if done:
            break

    # --------------------------------------------------------
    # Epsilon decay
    # --------------------------------------------------------

    agent.decay_epsilon()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    episode_length = step

    path_length = len(env.path)

    average_reward = total_reward / episode_length

    average_loss = (
        float(np.mean(losses))
        if losses
        else 0.0
    )

    episode_time = time.time() - episode_start

    reward_history.append(total_reward)
    success_history.append(1 if success else 0)
    steps_history.append(episode_length)

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    window = min(20, len(reward_history))

    moving_average_reward = float(
        np.mean(reward_history[-window:])
    )

    moving_success_rate = float(
        np.mean(success_history[-window:]) * 100
    )

    moving_average_steps = float(
        np.mean(steps_history[-window:])
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "Episode": episode,
        "Obstacle_Density": density,
        "Cumulative_Reward": total_reward,
        "Average_Reward": average_reward,
        "Training_Loss": average_loss,
        "Episode_Length": episode_length,
        "Path_Length": path_length,
        "Success": int(success),
        "Collision": int(collision),
        "Epsilon": agent.epsilon,
        "Episode_Time": episode_time,
        "Moving_Average_Reward": moving_average_reward,
        "Moving_Success_Rate": moving_success_rate,
        "Moving_Average_Steps": moving_average_steps
    })

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if moving_average_reward > best_moving_reward:

        best_moving_reward = moving_average_reward

        agent.save(BEST_MODEL_PATH)

    # --------------------------------------------------------
    # Target network update
    # --------------------------------------------------------

    if episode % TARGET_UPDATE_FREQUENCY == 0:

        agent.update_target_network()

    # --------------------------------------------------------
    # Progress display
    # --------------------------------------------------------

    if episode % 10 == 0 or episode == 1:

        print(
            f"Episode {episode:3d}/{NUM_EPISODES} | "
            f"Density {density:.2f} | "
            f"Reward {total_reward:8.2f} | "
            f"MA Reward {moving_average_reward:8.2f} | "
            f"Success {moving_success_rate:6.2f}% | "
            f"Steps {episode_length:3d} | "
            f"Loss {average_loss:.4f} | "
            f"Epsilon {agent.epsilon:.3f}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

agent.save(MODEL_PATH)


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

total_training_time = time.time() - start_time

average_training_reward = df["Cumulative_Reward"].mean()

average_episode_steps = df["Episode_Length"].mean()

average_training_loss = df["Training_Loss"].mean()

successful_episodes = df["Success"].sum()

success_rate = (
    successful_episodes / NUM_EPISODES
) * 100

collision_rate = (
    df["Collision"].sum() / NUM_EPISODES
) * 100

best_success_rate = df["Moving_Success_Rate"].max()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"Total Training Time       : {total_training_time:.2f} seconds")
print(f"Average Training Reward   : {average_training_reward:.2f}")
print(f"Average Episode Steps     : {average_episode_steps:.2f}")
print(f"Average Training Loss     : {average_training_loss:.4f}")

print(
    f"Successful Episodes       : "
    f"{successful_episodes}/{NUM_EPISODES}"
)

print(f"Overall Success Rate      : {success_rate:.2f}%")
print(f"Collision Rate            : {collision_rate:.2f}%")
print(f"Best 20-Episode Success   : {best_success_rate:.2f}%")

print(f"Final Epsilon             : {agent.epsilon:.3f}")

print()
print(f"Final Model               : {MODEL_PATH}")
print(f"Best Model                : {BEST_MODEL_PATH}")
print(f"Training CSV              : {CSV_PATH}")

print("=" * 70)