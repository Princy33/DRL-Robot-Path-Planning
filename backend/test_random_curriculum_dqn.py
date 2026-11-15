import os
import sys
import random
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

TEST_EPISODES = 50
GRID_SIZE = 20
MAX_STEPS = 150

OBSTACLE_DENSITY = 0.15
RANDOM_SEED = 5000

MODEL_PATH = "models/dqn_random_curriculum_model.pth"

CSV_PATH = "results/csv/dqn_random_curriculum_test_results.csv"


# ============================================================
# SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs("results/csv", exist_ok=True)


# ============================================================
# ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS,
    random_obstacles=True,
    obstacle_density=OBSTACLE_DENSITY
)


# ============================================================
# AGENT
# ============================================================

agent = DQNAgent(
    state_size=405,
    action_size=4,
    epsilon_start=0.0,
    epsilon_min=0.0
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("RANDOM CURRICULUM DQN - UNSEEN MAP TESTING")
print("=" * 70)

print(f"Model                  : {MODEL_PATH}")
print(f"Test Episodes          : {TEST_EPISODES}")
print(f"Grid Size              : {GRID_SIZE} x {GRID_SIZE}")
print(f"Obstacle Density      : {OBSTACLE_DENSITY:.2f}")
print(f"Maximum Steps          : {MAX_STEPS}")
print(f"Random Seed            : {RANDOM_SEED}")
print("=" * 70)

agent.load(MODEL_PATH)

# Completely greedy testing
agent.epsilon = 0.0


# ============================================================
# TESTING STORAGE
# ============================================================

results = []


# ============================================================
# TEST LOOP
# ============================================================

for episode in range(1, TEST_EPISODES + 1):

    episode_seed = RANDOM_SEED + episode

    state, info = env.reset(seed=episode_seed)

    total_reward = 0.0
    collision = False
    success = False

    # --------------------------------------------------------
    # Run episode
    # --------------------------------------------------------

    for step in range(1, MAX_STEPS + 1):

        action = agent.select_action(
            state,
            training=False
        )

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        state = next_state

        total_reward += reward

        if info.get("collision", False):
            collision = True

        if terminated and info.get("success", False):
            success = True

        if terminated or truncated:
            break

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    episode_length = step
    path_length = len(env.path)

    final_position = env.robot_position
    goal_position = env.goal_position

    final_distance = abs(
        final_position[0] - goal_position[0]
    ) + abs(
        final_position[1] - goal_position[1]
    )

    average_reward = total_reward / episode_length

    results.append({
        "Episode": episode,
        "Obstacle_Density": OBSTACLE_DENSITY,
        "Cumulative_Reward": total_reward,
        "Average_Reward": average_reward,
        "Episode_Length": episode_length,
        "Path_Length": path_length,
        "Success": int(success),
        "Collision": int(collision),
        "Final_Distance": final_distance
    })

    print(
        f"Episode {episode:2d}/{TEST_EPISODES} | "
        f"Reward {total_reward:8.2f} | "
        f"Steps {episode_length:3d} | "
        f"Success {'YES' if success else 'NO ':3s} | "
        f"Collision {'YES' if collision else 'NO ':3s} | "
        f"Distance {final_distance:2d}"
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

success_rate = df["Success"].mean() * 100

collision_rate = df["Collision"].mean() * 100

average_reward = df["Cumulative_Reward"].mean()

average_episode_reward = df["Average_Reward"].mean()

average_episode_length = df["Episode_Length"].mean()

average_path_length = df["Path_Length"].mean()

average_final_distance = df["Final_Distance"].mean()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("RANDOM CURRICULUM DQN TEST COMPLETED")
print("=" * 70)

print(f"Success Rate             : {success_rate:.2f}%")
print(f"Collision Rate           : {collision_rate:.2f}%")
print(f"Average Cumulative Reward: {average_reward:.2f}")
print(f"Average Reward / Step    : {average_episode_reward:.2f}")
print(f"Average Episode Length   : {average_episode_length:.2f}")
print(f"Average Path Length      : {average_path_length:.2f}")
print(f"Average Final Distance   : {average_final_distance:.2f}")

print()
print(f"Results CSV              : {CSV_PATH}")

print("=" * 70)