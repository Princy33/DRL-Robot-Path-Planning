import os
import time

import numpy as np
import pandas as pd

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_EPISODES = 500

GRID_SIZE = 20
MAX_STEPS = 150

BATCH_SIZE = 64

TARGET_UPDATE_FREQUENCY = 10

MODEL_PATH = "models/dqn_model.pth"
BEST_MODEL_PATH = "models/dqn_best_model.pth"

RESULTS_PATH = "results/csv/dqn_training_results.csv"


# ==========================================================
# CREATE REQUIRED DIRECTORIES
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/graphs", exist_ok=True)
os.makedirs("results/paths", exist_ok=True)


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_dqn():

    print("=" * 75)
    print("DEEP Q-NETWORK TRAINING")
    print("DRL ROBOT PATH PLANNING")
    print("=" * 75)

    # ------------------------------------------------------
    # Environment
    # ------------------------------------------------------

    env = RobotPathPlanningEnv(
        grid_size=GRID_SIZE,
        max_steps=MAX_STEPS
    )

    # ------------------------------------------------------
    # DQN Agent
    # ------------------------------------------------------

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY
    )

    # ------------------------------------------------------
    # Training Configuration
    # ------------------------------------------------------

    print("\nTraining Configuration:")
    print("-" * 55)
    print(f"Grid Size              : {GRID_SIZE} x {GRID_SIZE}")
    print(f"Number of Episodes     : {NUM_EPISODES}")
    print(f"Maximum Steps/Episode  : {MAX_STEPS}")
    print(f"Batch Size             : {BATCH_SIZE}")
    print(f"State Size             : {env.observation_space.shape[0]}")
    print(f"Action Size            : {env.action_space.n}")
    print(f"Device                 : {agent.device}")
    print(f"Initial Epsilon        : {agent.epsilon}")
    print("-" * 55)

    # ------------------------------------------------------
    # Training History
    # ------------------------------------------------------

    history = []

    best_success_rate = -1.0
    best_average_reward = -np.inf

    total_training_start = time.time()

    # ======================================================
    # EPISODE LOOP
    # ======================================================

    for episode in range(1, NUM_EPISODES + 1):

        episode_start = time.time()

        # --------------------------------------------------
        # Reset Environment
        # --------------------------------------------------

        state, info = env.reset(seed=episode)

        total_reward = 0.0
        losses = []

        done = False

        # --------------------------------------------------
        # STEP LOOP
        # --------------------------------------------------

        while not done:

            # Select action using epsilon-greedy policy
            action = agent.select_action(
                state,
                training=True
            )

            # Environment step
            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = terminated or truncated

            # Store experience
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Train neural network
            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

            # Move to next state
            state = next_state

            total_reward += reward

        # --------------------------------------------------
        # Epsilon Decay
        # --------------------------------------------------

        agent.decay_epsilon()

        # --------------------------------------------------
        # Target Network Update
        # --------------------------------------------------

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()

        # --------------------------------------------------
        # Episode Statistics
        # --------------------------------------------------

        episode_length = info["steps"]
        path_length = info["path_length"]

        success = int(info["success"])
        collision = int(info["collision"])

        average_loss = (
            float(np.mean(losses))
            if losses
            else 0.0
        )

        average_reward = (
            total_reward / max(episode_length, 1)
        )

        episode_time = time.time() - episode_start

        # --------------------------------------------------
        # Store Results
        # --------------------------------------------------

        history.append(
            {
                "Episode": episode,
                "Cumulative_Reward": total_reward,
                "Average_Reward": average_reward,
                "Training_Loss": average_loss,
                "Episode_Length": episode_length,
                "Path_Length": path_length,
                "Success": success,
                "Collision": collision,
                "Epsilon": agent.epsilon,
                "Episode_Time": episode_time
            }
        )

        # --------------------------------------------------
        # Calculate Recent Statistics
        # --------------------------------------------------

        recent_data = history[-20:]

        recent_rewards = [
            row["Cumulative_Reward"]
            for row in recent_data
        ]

        recent_success = [
            row["Success"]
            for row in recent_data
        ]

        recent_steps = [
            row["Episode_Length"]
            for row in recent_data
        ]

        moving_average_reward = float(
            np.mean(recent_rewards)
        )

        moving_success_rate = float(
            np.mean(recent_success) * 100
        )

        moving_average_steps = float(
            np.mean(recent_steps)
        )

        # --------------------------------------------------
        # Save Best Model
        # --------------------------------------------------

        if (
            moving_success_rate > best_success_rate
            or (
                moving_success_rate == best_success_rate
                and moving_average_reward > best_average_reward
            )
        ):

            best_success_rate = moving_success_rate
            best_average_reward = moving_average_reward

            agent.save(BEST_MODEL_PATH)

        # --------------------------------------------------
        # Progress Display
        # --------------------------------------------------

        if episode == 1 or episode % 10 == 0:

            print(
                f"Episode {episode:03d}/{NUM_EPISODES} | "
                f"Reward: {total_reward:8.2f} | "
                f"MA Reward: {moving_average_reward:8.2f} | "
                f"Loss: {average_loss:.4f} | "
                f"Steps: {episode_length:3d} | "
                f"Success: {moving_success_rate:5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # ======================================================
    # TRAINING COMPLETED
    # ======================================================

    total_training_time = (
        time.time() - total_training_start
    )

    # ------------------------------------------------------
    # Convert History to DataFrame
    # ------------------------------------------------------

    results_df = pd.DataFrame(history)

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # ------------------------------------------------------
    # Save Final Model
    # ------------------------------------------------------

    agent.save(MODEL_PATH)

    # ------------------------------------------------------
    # Final Statistics
    # ------------------------------------------------------

    overall_success_rate = (
        results_df["Success"].mean() * 100
    )

    overall_collision_rate = (
        results_df["Collision"].mean() * 100
    )

    average_reward = (
        results_df["Cumulative_Reward"].mean()
    )

    average_steps = (
        results_df["Episode_Length"].mean()
    )

    average_loss = (
        results_df["Training_Loss"].mean()
    )

    successful_episodes = int(
        results_df["Success"].sum()
    )

    # ======================================================
    # FINAL REPORT
    # ======================================================

    print("\n" + "=" * 75)
    print("TRAINING COMPLETED")
    print("=" * 75)

    print(
        f"Total Training Time    : "
        f"{total_training_time:.2f} seconds"
    )

    print(
        f"Average Reward         : "
        f"{average_reward:.2f}"
    )

    print(
        f"Average Episode Steps  : "
        f"{average_steps:.2f}"
    )

    print(
        f"Average Training Loss  : "
        f"{average_loss:.4f}"
    )

    print(
        f"Successful Episodes    : "
        f"{successful_episodes}/{NUM_EPISODES}"
    )

    print(
        f"Overall Success Rate   : "
        f"{overall_success_rate:.2f}%"
    )

    print(
        f"Collision Rate         : "
        f"{overall_collision_rate:.2f}%"
    )

    print(
        f"Best 20-Episode Success: "
        f"{best_success_rate:.2f}%"
    )

    print(
        f"\nFinal Model saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nBest Model saved to:"
        f"\n{BEST_MODEL_PATH}"
    )

    print(
        f"\nTraining Results saved to:"
        f"\n{RESULTS_PATH}"
    )

    print("=" * 75)

    env.close()

    return results_df


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train_dqn()