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

REPLAY_CAPACITY = 100000

RANDOM_SEED = 2026


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = (
    "models/ddqn_random_curriculum_model.pth"
)

BEST_MODEL_PATH = (
    "models/ddqn_random_curriculum_best_model.pth"
)

CSV_PATH = (
    "results/csv/"
    "ddqn_random_curriculum_training_results.csv"
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results/csv",
    exist_ok=True
)


# ============================================================
# CURRICULUM FUNCTION
# ============================================================

def get_obstacle_density(episode):

    if episode <= 150:

        return 0.05

    elif episode <= 300:

        return 0.08

    elif episode <= 400:

        return 0.12

    else:

        return 0.15


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS,
    random_obstacles=True,
    obstacle_density=0.05
)


# ============================================================
# CREATE DDQN AGENT
# ============================================================

agent = DDQNAgent(
    state_size=405,
    action_size=4,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    replay_capacity=REPLAY_CAPACITY,
    batch_size=BATCH_SIZE
)


# ============================================================
# STORAGE
# ============================================================

results = []

reward_history = []

success_history = []

steps_history = []


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_moving_reward = -float("inf")


# ============================================================
# TRAINING TIMER
# ============================================================

training_start_time = time.time()


# ============================================================
# TRAINING HEADER
# ============================================================

print("=" * 70)

print(
    "CURRICULUM-BASED RANDOM DDQN TRAINING"
)

print("=" * 70)

print(
    f"Grid Size              : "
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    f"Maximum Steps          : "
    f"{MAX_STEPS}"
)

print(
    f"Training Episodes      : "
    f"{NUM_EPISODES}"
)

print(
    f"Batch Size             : "
    f"{BATCH_SIZE}"
)

print(
    f"Learning Rate          : "
    f"{LEARNING_RATE}"
)

print(
    f"Gamma                  : "
    f"{GAMMA}"
)

print(
    f"Initial Epsilon        : "
    f"{EPSILON_START}"
)

print(
    f"Minimum Epsilon        : "
    f"{EPSILON_MIN}"
)

print(
    f"Epsilon Decay          : "
    f"{EPSILON_DECAY}"
)

print(
    f"Target Update          : "
    f"Every {TARGET_UPDATE_FREQUENCY} episodes"
)

print(
    f"Random Seed            : "
    f"{RANDOM_SEED}"
)

print("=" * 70)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

for episode in range(
    1,
    NUM_EPISODES + 1
):

    episode_start_time = time.time()


    # ========================================================
    # CURRICULUM DIFFICULTY
    # ========================================================

    density = get_obstacle_density(
        episode
    )

    env.obstacle_density = density


    # ========================================================
    # EPISODE SEED
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

    losses = []

    success = False

    collision = False


    # ========================================================
    # EPISODE LOOP
    # ========================================================

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        # ----------------------------------------------------
        # SELECT ACTION
        # ----------------------------------------------------

        action = agent.select_action(
            state,
            training=True
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
        ) = env.step(action)


        # ----------------------------------------------------
        # DONE FLAG
        # ----------------------------------------------------

        done = (
            terminated or truncated
        )


        # ----------------------------------------------------
        # STORE EXPERIENCE
        # ----------------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )


        # ----------------------------------------------------
        # TRAIN DDQN
        # ----------------------------------------------------

        loss = agent.train_step()


        if loss is not None:

            losses.append(
                loss
            )


        # ----------------------------------------------------
        # UPDATE STATE
        # ----------------------------------------------------

        state = next_state


        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        total_reward += reward


        # ----------------------------------------------------
        # SUCCESS
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
        # COLLISION
        # ----------------------------------------------------

        if info.get(
            "collision",
            False
        ):

            collision = True


        # ----------------------------------------------------
        # END EPISODE
        # ----------------------------------------------------

        if done:

            break


    # ========================================================
    # EPSILON DECAY
    # ========================================================

    agent.decay_epsilon()


    # ========================================================
    # TARGET NETWORK UPDATE
    # ========================================================

    if (
        episode
        % TARGET_UPDATE_FREQUENCY
        == 0
    ):

        agent.update_target_network()


    # ========================================================
    # EPISODE METRICS
    # ========================================================

    episode_length = step

    path_length = len(
        env.path
    )


    # --------------------------------------------------------
    # AVERAGE REWARD
    # --------------------------------------------------------

    average_reward = (
        total_reward
        / episode_length
    )


    # --------------------------------------------------------
    # AVERAGE LOSS
    # --------------------------------------------------------

    if losses:

        average_loss = float(
            np.mean(losses)
        )

    else:

        average_loss = 0.0


    # --------------------------------------------------------
    # EPISODE TIME
    # --------------------------------------------------------

    episode_time = (
        time.time()
        - episode_start_time
    )


    # ========================================================
    # HISTORY
    # ========================================================

    reward_history.append(
        total_reward
    )

    success_history.append(
        1 if success else 0
    )

    steps_history.append(
        episode_length
    )


    # ========================================================
    # MOVING WINDOW
    # ========================================================

    window = min(
        20,
        len(reward_history)
    )


    # --------------------------------------------------------
    # MOVING AVERAGE REWARD
    # --------------------------------------------------------

    moving_average_reward = float(
        np.mean(
            reward_history[-window:]
        )
    )


    # --------------------------------------------------------
    # MOVING SUCCESS RATE
    # --------------------------------------------------------

    moving_success_rate = float(
        np.mean(
            success_history[-window:]
        )
        * 100
    )


    # --------------------------------------------------------
    # MOVING AVERAGE STEPS
    # --------------------------------------------------------

    moving_average_steps = float(
        np.mean(
            steps_history[-window:]
        )
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "Episode":
            episode,

        "Obstacle_Density":
            density,

        "Cumulative_Reward":
            total_reward,

        "Average_Reward":
            average_reward,

        "Training_Loss":
            average_loss,

        "Episode_Length":
            episode_length,

        "Path_Length":
            path_length,

        "Success":
            int(success),

        "Collision":
            int(collision),

        "Epsilon":
            agent.epsilon,

        "Episode_Time":
            episode_time,

        "Moving_Average_Reward":
            moving_average_reward,

        "Moving_Success_Rate":
            moving_success_rate,

        "Moving_Average_Steps":
            moving_average_steps
    })


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        moving_average_reward
        > best_moving_reward
    ):

        best_moving_reward = (
            moving_average_reward
        )

        agent.save(
            BEST_MODEL_PATH
        )


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if (
        episode % 10 == 0
        or episode == 1
    ):

        print(

            f"Episode "
            f"{episode:3d}/"
            f"{NUM_EPISODES} | "

            f"Density "
            f"{density:.2f} | "

            f"Reward "
            f"{total_reward:8.2f} | "

            f"MA Reward "
            f"{moving_average_reward:8.2f} | "

            f"Success "
            f"{moving_success_rate:6.2f}% | "

            f"Steps "
            f"{episode_length:3d} | "

            f"Loss "
            f"{average_loss:.4f} | "

            f"Epsilon "
            f"{agent.epsilon:.3f}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

agent.save(
    MODEL_PATH
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

total_training_time = (
    time.time()
    - training_start_time
)


average_training_reward = (
    df[
        "Cumulative_Reward"
    ].mean()
)


average_episode_steps = (
    df[
        "Episode_Length"
    ].mean()
)


average_training_loss = (
    df[
        "Training_Loss"
    ].mean()
)


successful_episodes = int(
    df[
        "Success"
    ].sum()
)


success_rate = (
    successful_episodes
    / NUM_EPISODES
) * 100


collision_rate = (
    df[
        "Collision"
    ].mean()
) * 100


best_success_rate = (
    df[
        "Moving_Success_Rate"
    ].max()
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "DDQN TRAINING COMPLETED"
)

print("=" * 70)


print(
    f"Total Training Time       : "
    f"{total_training_time:.2f} seconds"
)


print(
    f"Average Training Reward   : "
    f"{average_training_reward:.2f}"
)


print(
    f"Average Episode Steps     : "
    f"{average_episode_steps:.2f}"
)


print(
    f"Average Training Loss     : "
    f"{average_training_loss:.4f}"
)


print(
    f"Successful Episodes       : "
    f"{successful_episodes}/"
    f"{NUM_EPISODES}"
)


print(
    f"Overall Success Rate      : "
    f"{success_rate:.2f}%"
)


print(
    f"Collision Rate            : "
    f"{collision_rate:.2f}%"
)


print(
    f"Best 20-Episode Success   : "
    f"{best_success_rate:.2f}%"
)


print(
    f"Final Epsilon             : "
    f"{agent.epsilon:.3f}"
)


print()

print(
    f"Final Model               : "
    f"{MODEL_PATH}"
)


print(
    f"Best Model                : "
    f"{BEST_MODEL_PATH}"
)


print(
    f"Training CSV              : "
    f"{CSV_PATH}"
)


print("=" * 70)