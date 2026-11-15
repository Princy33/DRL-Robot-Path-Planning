import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from backend.environment import RobotPathPlanningEnv
from backend.ddqn_agent import DDQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

TEST_EPISODES = 50
GRID_SIZE = 20
MAX_STEPS = 150

MODEL_PATH = "models/ddqn_model.pth"

RESULTS_DIR = "results/csv"
RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "ddqn_test_results.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS
)


# ============================================================
# AGENT
# ============================================================

agent = DDQNAgent(
    state_size=405,
    action_size=4
)

agent.load(MODEL_PATH)

# Greedy evaluation
agent.epsilon = 0.0


print("=" * 70)
print("DDQN MODEL EVALUATION")
print("=" * 70)

print(f"Test Episodes : {TEST_EPISODES}")
print(f"Grid Size     : {GRID_SIZE} x {GRID_SIZE}")
print(f"Max Steps     : {MAX_STEPS}")
print(f"Epsilon       : {agent.epsilon}")
print()


# ============================================================
# TESTING
# ============================================================

results = []

successful_episodes = 0
collision_episodes = 0

for episode in range(1, TEST_EPISODES + 1):

    # Same deterministic environment used for DQN testing
    state, info = env.reset(seed=42)

    total_reward = 0.0
    steps = 0

    success = False
    collision = False

    start_time = time.time()

    while steps < MAX_STEPS:

        action = agent.select_action(
            state,
            training=False
        )

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        steps += 1

        state = next_state

        if terminated or truncated:

            success = info.get(
                "success",
                False
            )

            collision = info.get(
                "collision",
                False
            )

            break

    elapsed_time = time.time() - start_time

    if success:
        successful_episodes += 1

    if collision:
        collision_episodes += 1

    path_length = len(env.path)

    average_reward = (
        total_reward / steps
        if steps > 0
        else 0.0
    )

    results.append({
        "Episode": episode,
        "Cumulative Reward": total_reward,
        "Average Reward": average_reward,
        "Episode Length": steps,
        "Path Length": path_length,
        "Success": int(success),
        "Collision": int(collision),
        "Testing Time": elapsed_time
    })

    print(
        f"Episode {episode:03d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Steps: {steps:3d} | "
        f"Success: {'YES' if success else 'NO'}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

success_rate = (
    successful_episodes / TEST_EPISODES
) * 100

collision_rate = (
    collision_episodes / TEST_EPISODES
) * 100

avg_reward = df[
    "Cumulative Reward"
].mean()

avg_step_reward = df[
    "Average Reward"
].mean()

avg_episode_length = df[
    "Episode Length"
].mean()

avg_path_length = df[
    "Path Length"
].mean()


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("FINAL DDQN TEST RESULTS")
print("=" * 70)

print(
    f"Successful Episodes : "
    f"{successful_episodes}/{TEST_EPISODES}"
)

print(
    f"Success Rate        : "
    f"{success_rate:.2f}%"
)

print(
    f"Collision Rate      : "
    f"{collision_rate:.2f}%"
)

print(
    f"Average Reward      : "
    f"{avg_reward:.2f}"
)

print(
    f"Average Reward/Step : "
    f"{avg_step_reward:.2f}"
)

print(
    f"Average Episode Len : "
    f"{avg_episode_length:.2f}"
)

print(
    f"Average Path Length : "
    f"{avg_path_length:.2f}"
)

print()
print("Results saved to:")
print(RESULT_FILE)

print("=" * 70)