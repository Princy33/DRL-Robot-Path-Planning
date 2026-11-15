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

NUM_EPISODES = 500
GRID_SIZE = 20
MAX_STEPS = 150
BATCH_SIZE = 64

TARGET_UPDATE_FREQUENCY = 10

MODEL_DIR = "models"
RESULTS_DIR = "results/csv"

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ddqn_model.pth"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ddqn_best_model.pth"
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "ddqn_training_results.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS
)


# ============================================================
# DDQN AGENT
# ============================================================

agent = DDQNAgent(
    state_size=405,
    action_size=4,
    batch_size=BATCH_SIZE
)


# ============================================================
# STORAGE
# ============================================================

training_results = []

rewards_history = []
success_history = []
steps_history = []

best_success_rate = -1
best_reward = -float("inf")


# ============================================================
# TRAINING START
# ============================================================

training_start = time.time()

print("=" * 70)
print("DOUBLE DQN TRAINING")
print("=" * 70)

print(f"Episodes          : {NUM_EPISODES}")
print(f"Grid Size         : {GRID_SIZE} x {GRID_SIZE}")
print(f"Maximum Steps     : {MAX_STEPS}")
print(f"Batch Size        : {BATCH_SIZE}")
print(f"Target Update     : Every {TARGET_UPDATE_FREQUENCY} episodes")
print(f"Device            : {agent.device}")
print("=" * 70)
print()


# ============================================================
# TRAINING LOOP
# ============================================================

for episode in range(1, NUM_EPISODES + 1):

    episode_start = time.time()

    state, info = env.reset(seed=episode)

    total_reward = 0.0
    episode_loss = []
    steps = 0

    success = False
    collision = False

    while steps < MAX_STEPS:

        # ----------------------------------------------
        # Select action using epsilon-greedy policy
        # ----------------------------------------------

        action = agent.select_action(
            state,
            training=True
        )

        # ----------------------------------------------
        # Environment step
        # ----------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        # ----------------------------------------------
        # Store experience
        # ----------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # ----------------------------------------------
        # Train DDQN
        # ----------------------------------------------

        loss = agent.train_step()

        if loss is not None:
            episode_loss.append(loss)

        # ----------------------------------------------
        # Update state
        # ----------------------------------------------

        state = next_state

        total_reward += reward
        steps += 1

        # ----------------------------------------------
        # Episode termination
        # ----------------------------------------------

        if done:

            success = info.get(
                "success",
                False
            )

            collision = info.get(
                "collision",
                False
            )

            break

    # ========================================================
    # UPDATE TARGET NETWORK
    # ========================================================

    if episode % TARGET_UPDATE_FREQUENCY == 0:

        agent.update_target_network()

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    agent.decay_epsilon()

    # ========================================================
    # EPISODE METRICS
    # ========================================================

    average_reward = (
        total_reward / steps
        if steps > 0
        else 0.0
    )

    average_loss = (
        np.mean(episode_loss)
        if episode_loss
        else 0.0
    )

    path_length = len(env.path)

    episode_time = time.time() - episode_start

    rewards_history.append(total_reward)
    success_history.append(int(success))
    steps_history.append(steps)

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    window = min(20, len(rewards_history))

    moving_average_reward = np.mean(
        rewards_history[-window:]
    )

    moving_success_rate = (
        np.mean(success_history[-window:]) * 100
    )

    moving_average_steps = np.mean(
        steps_history[-window:]
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        moving_success_rate > best_success_rate
        or (
            moving_success_rate == best_success_rate
            and moving_average_reward > best_reward
        )
    ):

        best_success_rate = moving_success_rate
        best_reward = moving_average_reward

        agent.save(BEST_MODEL_PATH)

    # ========================================================
    # SAVE RESULT
    # ========================================================

    training_results.append({

        "Episode": episode,

        "Cumulative_Reward": total_reward,

        "Average_Reward": average_reward,

        "Training_Loss": average_loss,

        "Episode_Length": steps,

        "Path_Length": path_length,

        "Success": int(success),

        "Collision": int(collision),

        "Epsilon": agent.epsilon,

        "Episode_Time": episode_time,

        "Moving_Average_Reward": moving_average_reward,

        "Moving_Success_Rate": moving_success_rate,

        "Moving_Average_Steps": moving_average_steps
    })

    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if episode == 1 or episode % 10 == 0:

        print(
            f"Episode {episode:03d} | "
            f"Reward: {total_reward:8.2f} | "
            f"MA Reward: {moving_average_reward:8.2f} | "
            f"Loss: {average_loss:7.4f} | "
            f"Steps: {steps:3d} | "
            f"Success: {'YES' if success else 'NO '} | "
            f"MA Success: {moving_success_rate:6.2f}% | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

agent.save(FINAL_MODEL_PATH)


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(training_results)

df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

total_training_time = (
    time.time() - training_start
)

successful_episodes = sum(
    success_history
)

collision_episodes = sum(
    1 for result in training_results
    if result["Collision"] == 1
)

overall_success_rate = (
    successful_episodes / NUM_EPISODES
) * 100

collision_rate = (
    collision_episodes / NUM_EPISODES
) * 100

average_training_reward = np.mean(
    rewards_history
)

average_episode_steps = np.mean(
    steps_history
)

average_training_loss = np.mean(
    [
        result["Training_Loss"]
        for result in training_results
    ]
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("DDQN TRAINING COMPLETED")
print("=" * 70)

print(
    f"Total Training Time : "
    f"{total_training_time:.2f} seconds"
)

print(
    f"Average Reward      : "
    f"{average_training_reward:.2f}"
)

print(
    f"Average Episode Steps : "
    f"{average_episode_steps:.2f}"
)

print(
    f"Average Training Loss : "
    f"{average_training_loss:.4f}"
)

print(
    f"Successful Episodes : "
    f"{successful_episodes}/{NUM_EPISODES}"
)

print(
    f"Overall Success Rate : "
    f"{overall_success_rate:.2f}%"
)

print(
    f"Collision Rate      : "
    f"{collision_rate:.2f}%"
)

print(
    f"Best 20-Episode Success : "
    f"{best_success_rate:.2f}%"
)

print()
print("Final Model:")
print(FINAL_MODEL_PATH)

print()
print("Best Model:")
print(BEST_MODEL_PATH)

print()
print("Training CSV:")
print(RESULT_FILE)

print("=" * 70)
