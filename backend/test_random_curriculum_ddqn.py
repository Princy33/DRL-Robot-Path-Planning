import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch

# ============================================================
# BACKEND IMPORT PATH
# ============================================================

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from environment import RobotPathPlanningEnv
from ddqn_agent import DDQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

TEST_EPISODES = 50

GRID_SIZE = 20
MAX_STEPS = 150

OBSTACLE_DENSITY = 0.15

RANDOM_SEED = 5000


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    "models/ddqn_random_curriculum_model.pth"
)


# ============================================================
# RESULT PATHS
# ============================================================

CSV_PATH = (
    "results/csv/"
    "ddqn_random_curriculum_test_results.csv"
)

SUMMARY_PATH = (
    "results/csv/"
    "ddqn_random_curriculum_test_summary.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "results/csv",
    exist_ok=True
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS,
    random_obstacles=True,
    obstacle_density=OBSTACLE_DENSITY
)


# ============================================================
# CREATE DDQN AGENT
# ============================================================

agent = DDQNAgent(
    state_size=405,
    action_size=4,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=0.05,
    epsilon_min=0.05,
    epsilon_decay=1.0,
    replay_capacity=100000,
    batch_size=64
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("=" * 70)
    print("ERROR: MODEL NOT FOUND")
    print("=" * 70)

    print(
        f"Expected model:"
        f" {MODEL_PATH}"
    )

    print()
    print(
        "Please check that the DDQN training "
        "was completed successfully."
    )

    print("=" * 70)

    sys.exit(1)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

agent.load(
    MODEL_PATH
)


# ============================================================
# GREEDY EVALUATION
# ============================================================

agent.epsilon = 0.0


# ============================================================
# TESTING HEADER
# ============================================================

print()

print("=" * 70)

print(
    "RANDOM CURRICULUM DDQN - "
    "UNSEEN MAP TESTING"
)

print("=" * 70)

print(
    f"Test Episodes          : "
    f"{TEST_EPISODES}"
)

print(
    f"Grid Size              : "
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    f"Obstacle Density       : "
    f"{OBSTACLE_DENSITY}"
)

print(
    f"Maximum Steps          : "
    f"{MAX_STEPS}"
)

print(
    f"Random Seed            : "
    f"{RANDOM_SEED}"
)

print(
    f"Model                  : "
    f"{MODEL_PATH}"
)

print("=" * 70)


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# TESTING TIMER
# ============================================================

testing_start_time = time.time()


# ============================================================
# TESTING LOOP
# ============================================================

for episode in range(
    1,
    TEST_EPISODES + 1
):

    episode_start_time = time.time()


    # ========================================================
    # UNIQUE SEED
    # ========================================================

    episode_seed = (
        RANDOM_SEED + episode
    )


    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    state, info = env.reset(
        seed=episode_seed
    )


    # ========================================================
    # EPISODE VARIABLES
    # ========================================================

    total_reward = 0.0

    success = False

    collision = False


    # ========================================================
    # RUN EPISODE
    # ========================================================

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        # ----------------------------------------------------
        # SELECT GREEDY ACTION
        # ----------------------------------------------------

        action = agent.select_action(
            state,
            training=False
        )


        # ----------------------------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        # ----------------------------------------------------
        # UPDATE REWARD
        # ----------------------------------------------------

        total_reward += reward


        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        state = next_state


        # ----------------------------------------------------
        # SUCCESS CHECK
        # ----------------------------------------------------

        if (
            terminated
            and
            info.get(
                "success",
                False
            )
        ):

            success = True


        # ----------------------------------------------------
        # COLLISION CHECK
        # ----------------------------------------------------

        if info.get(
            "collision",
            False
        ):

            collision = True


        # ----------------------------------------------------
        # END EPISODE
        # ----------------------------------------------------

        if (
            terminated
            or truncated
        ):

            break


    # ========================================================
    # EPISODE METRICS
    # ========================================================

    episode_length = step

    path_length = len(
        env.path
    )


    # ========================================================
    # FINAL POSITION
    # ========================================================

    robot_position = (
        env.robot_position
    )

    goal_position = (
        env.goal_position
    )


    # ========================================================
    # FINAL MANHATTAN DISTANCE
    # ========================================================

    final_distance = (
        abs(
            robot_position[0]
            -
            goal_position[0]
        )
        +
        abs(
            robot_position[1]
            -
            goal_position[1]
        )
    )


    # ========================================================
    # AVERAGE REWARD PER STEP
    # ========================================================

    average_reward_per_step = (
        total_reward
        /
        episode_length
    )


    # ========================================================
    # EPISODE TESTING TIME
    # ========================================================

    episode_time = (
        time.time()
        -
        episode_start_time
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "Episode":
            episode,

        "Obstacle_Density":
            OBSTACLE_DENSITY,

        "Seed":
            episode_seed,

        "Cumulative_Reward":
            total_reward,

        "Average_Reward_Per_Step":
            average_reward_per_step,

        "Episode_Length":
            episode_length,

        "Path_Length":
            path_length,

        "Success":
            int(success),

        "Collision":
            int(collision),

        "Final_Distance":
            final_distance,

        "Episode_Time":
            episode_time
    })


    # ========================================================
    # PRINT EPISODE RESULT
    # ========================================================

    print(

        f"Episode "
        f"{episode:2d}/"
        f"{TEST_EPISODES} | "

        f"Reward "
        f"{total_reward:8.2f} | "

        f"Steps "
        f"{episode_length:3d} | "

        f"Success "
        f"{'YES' if success else 'NO ':3s} | "

        f"Collision "
        f"{'YES' if collision else 'NO ':3s} | "

        f"Distance "
        f"{final_distance:2d}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# CALCULATE SUMMARY METRICS
# ============================================================

success_rate = (
    df["Success"].mean()
    *
    100
)


collision_rate = (
    df["Collision"].mean()
    *
    100
)


average_reward = (
    df["Cumulative_Reward"].mean()
)


average_reward_per_step = (
    df[
        "Average_Reward_Per_Step"
    ].mean()
)


average_episode_length = (
    df["Episode_Length"].mean()
)


average_path_length = (
    df["Path_Length"].mean()
)


average_final_distance = (
    df["Final_Distance"].mean()
)


average_test_time = (
    df["Episode_Time"].mean()
)


total_testing_time = (
    time.time()
    -
    testing_start_time
)


# ============================================================
# CREATE SUMMARY DATAFRAME
# ============================================================

summary = pd.DataFrame({

    "Model": [
        "DDQN"
    ],

    "Obstacle_Density": [
        OBSTACLE_DENSITY
    ],

    "Test_Episodes": [
        TEST_EPISODES
    ],

    "Success_Rate_Percent": [
        success_rate
    ],

    "Collision_Rate_Percent": [
        collision_rate
    ],

    "Average_Cumulative_Reward": [
        average_reward
    ],

    "Average_Reward_Per_Step": [
        average_reward_per_step
    ],

    "Average_Episode_Length": [
        average_episode_length
    ],

    "Average_Path_Length": [
        average_path_length
    ],

    "Average_Final_Distance": [
        average_final_distance
    ],

    "Average_Test_Time_Seconds": [
        average_test_time
    ],

    "Total_Testing_Time_Seconds": [
        total_testing_time
    ]
})


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 70)

print(
    "RANDOM CURRICULUM DDQN "
    "TESTING COMPLETED"
)

print("=" * 70)

print(
    f"Test Episodes              : "
    f"{TEST_EPISODES}"
)

print(
    f"Obstacle Density           : "
    f"{OBSTACLE_DENSITY:.2f}"
)

print(
    f"Success Rate               : "
    f"{success_rate:.2f}%"
)

print(
    f"Collision Rate             : "
    f"{collision_rate:.2f}%"
)

print(
    f"Average Cumulative Reward  : "
    f"{average_reward:.2f}"
)

print(
    f"Average Reward / Step      : "
    f"{average_reward_per_step:.2f}"
)

print(
    f"Average Episode Length     : "
    f"{average_episode_length:.2f}"
)

print(
    f"Average Path Length        : "
    f"{average_path_length:.2f}"
)

print(
    f"Average Final Distance     : "
    f"{average_final_distance:.2f}"
)

print(
    f"Average Test Time          : "
    f"{average_test_time:.4f} seconds"
)

print(
    f"Total Testing Time         : "
    f"{total_testing_time:.2f} seconds"
)

print()

print(
    f"Detailed CSV               : "
    f"{CSV_PATH}"
)

print(
    f"Summary CSV                : "
    f"{SUMMARY_PATH}"
)

print("=" * 70)