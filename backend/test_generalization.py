import os
import sys
import time
import numpy as np
import pandas as pd
import torch

# Allow importing files from backend
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent


# =========================================================
# SETTINGS
# =========================================================

TEST_EPISODES = 50

GRID_SIZE = 20

MAX_STEPS = 150

OBSTACLE_DENSITY = 0.18

RANDOM_SEED = 2026

DQN_MODEL_PATH = "models/dqn_model.pth"

DDQN_MODEL_PATH = "models/ddqn_model.pth"

OUTPUT_DIRECTORY = "results/csv"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "generalization_results.csv"
)


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD DQN
# =========================================================

def load_dqn():

    agent = DQNAgent(
        state_size=405,
        action_size=4
    )

    agent.load(
        DQN_MODEL_PATH
    )

    agent.epsilon = 0.0

    return agent


# =========================================================
# LOAD DDQN
# =========================================================

def load_ddqn():

    agent = DDQNAgent(
        state_size=405,
        action_size=4
    )

    agent.load(
        DDQN_MODEL_PATH
    )

    agent.epsilon = 0.0

    return agent


# =========================================================
# TEST ONE MODEL
# =========================================================

def evaluate_model(
    model_name,
    agent,
    seeds
):

    results = []

    successful_episodes = 0

    collision_episodes = 0

    total_reward = 0.0

    total_steps = 0

    total_path_length = 0

    total_testing_time = 0.0

    print()
    print("=" * 60)
    print(
        f"{model_name} GENERALIZATION TEST"
    )
    print("=" * 60)

    for episode in range(
        TEST_EPISODES
    ):

        # -------------------------------------------------
        # Create a NEW random environment
        # -------------------------------------------------

        env = RobotPathPlanningEnv(
            grid_size=GRID_SIZE,
            max_steps=MAX_STEPS,
            random_obstacles=True,
            obstacle_density=OBSTACLE_DENSITY
        )

        # -------------------------------------------------
        # Reset using a different seed
        # -------------------------------------------------

        seed = seeds[episode]

        state, info = env.reset(
            seed=seed
        )

        done = False

        episode_reward = 0.0

        episode_steps = 0

        episode_collision = False

        start_time = time.time()

        # -------------------------------------------------
        # Greedy evaluation
        # -------------------------------------------------

        while not done:

            action = agent.select_action(
                state
            )

            next_state, reward, terminated, truncated, step_info = env.step(
                action
            )

            state = next_state

            episode_reward += reward

            episode_steps += 1

            if step_info.get(
                "collision",
                False
            ):

                episode_collision = True

            done = (
                terminated
                or
                truncated
            )

        testing_time = (
            time.time() -
            start_time
        )

        # -------------------------------------------------
        # Episode result
        # -------------------------------------------------

        success = bool(
            step_info.get(
                "success",
                False
            )
        )

        path_length = int(
            step_info.get(
                "path_length",
                episode_steps
            )
        )

        final_distance = int(
            step_info.get(
                "distance_to_goal",
                0
            )
        )

        if success:

            successful_episodes += 1

        if episode_collision:

            collision_episodes += 1

        total_reward += (
            episode_reward
        )

        total_steps += (
            episode_steps
        )

        total_path_length += (
            path_length
        )

        total_testing_time += (
            testing_time
        )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append(
            {
                "Model": model_name,
                "Episode": episode + 1,
                "Seed": seed,
                "Obstacle_Density": OBSTACLE_DENSITY,
                "Number_of_Obstacles": len(
                    env.obstacles
                ),
                "Cumulative_Reward": episode_reward,
                "Episode_Length": episode_steps,
                "Path_Length": path_length,
                "Success": int(success),
                "Collision": int(
                    episode_collision
                ),
                "Final_Distance": final_distance,
                "Testing_Time": testing_time
            }
        )

        # -------------------------------------------------
        # Print progress
        # -------------------------------------------------

        print(
            f"Episode {episode + 1:03d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Steps: {episode_steps:3d} | "
            f"Distance: {final_distance:2d} | "
            f"Success: "
            f"{'YES' if success else 'NO'}"
        )

        env.close()

    # =====================================================
    # SUMMARY
    # =====================================================

    success_rate = (
        successful_episodes /
        TEST_EPISODES
    ) * 100.0

    collision_rate = (
        collision_episodes /
        TEST_EPISODES
    ) * 100.0

    average_reward = (
        total_reward /
        TEST_EPISODES
    )

    average_steps = (
        total_steps /
        TEST_EPISODES
    )

    average_path_length = (
        total_path_length /
        TEST_EPISODES
    )

    average_testing_time = (
        total_testing_time /
        TEST_EPISODES
    )

    print()
    print(
        f"{model_name} FINAL GENERALIZATION RESULTS"
    )

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
        f"{average_reward:.2f}"
    )

    print(
        f"Average Episode Len : "
        f"{average_steps:.2f}"
    )

    print(
        f"Average Path Length : "
        f"{average_path_length:.2f}"
    )

    print(
        f"Average Test Time   : "
        f"{average_testing_time:.4f} sec"
    )

    return results


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print(
        "DRL ROBOT PATH PLANNING"
    )
    print(
        "UNSEEN RANDOM OBSTACLE GENERALIZATION TEST"
    )
    print("=" * 60)

    print()
    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Test Episodes       : {TEST_EPISODES}"
    )

    print(
        f"Grid Size           : "
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        f"Max Steps           : {MAX_STEPS}"
    )

    print(
        f"Obstacle Density    : "
        f"{OBSTACLE_DENSITY}"
    )

    print()

    # -----------------------------------------------------
    # Check model files
    # -----------------------------------------------------

    if not os.path.exists(
        DQN_MODEL_PATH
    ):

        print(
            "ERROR: DQN model not found:"
        )

        print(
            DQN_MODEL_PATH
        )

        return

    if not os.path.exists(
        DDQN_MODEL_PATH
    ):

        print(
            "ERROR: DDQN model not found:"
        )

        print(
            DDQN_MODEL_PATH
        )

        return

    # -----------------------------------------------------
    # Create reproducible test seeds
    # -----------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    seeds = rng.integers(
        0,
        1_000_000,
        size=TEST_EPISODES
    ).tolist()

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print(
        "Loading DQN model..."
    )

    dqn_agent = load_dqn()

    print(
        "DQN model loaded successfully."
    )

    print()

    print(
        "Loading DDQN model..."
    )

    ddqn_agent = load_ddqn()

    print(
        "DDQN model loaded successfully."
    )

    # -----------------------------------------------------
    # Evaluate DQN
    # -----------------------------------------------------

    dqn_results = evaluate_model(
        "DQN",
        dqn_agent,
        seeds
    )

    # -----------------------------------------------------
    # Evaluate DDQN
    # -----------------------------------------------------

    ddqn_results = evaluate_model(
        "DDQN",
        ddqn_agent,
        seeds
    )

    # -----------------------------------------------------
    # Combine results
    # -----------------------------------------------------

    all_results = (
        dqn_results +
        ddqn_results
    )

    results_df = pd.DataFrame(
        all_results
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # =====================================================
    # FINAL COMPARISON
    # =====================================================

    print()
    print("=" * 60)
    print(
        "DQN vs DDQN GENERALIZATION COMPARISON"
    )
    print("=" * 60)

    comparison = []

    for model_name in [
        "DQN",
        "DDQN"
    ]:

        model_data = results_df[
            results_df["Model"] ==
            model_name
        ]

        comparison.append(
            {
                "Model": model_name,

                "Success Rate (%)":
                    model_data[
                        "Success"
                    ].mean() * 100,

                "Collision Rate (%)":
                    model_data[
                        "Collision"
                    ].mean() * 100,

                "Average Reward":
                    model_data[
                        "Cumulative_Reward"
                    ].mean(),

                "Average Episode Length":
                    model_data[
                        "Episode_Length"
                    ].mean(),

                "Average Path Length":
                    model_data[
                        "Path_Length"
                    ].mean(),

                "Average Final Distance":
                    model_data[
                        "Final_Distance"
                    ].mean()
            }
        )

    comparison_df = pd.DataFrame(
        comparison
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save comparison
    # -----------------------------------------------------

    comparison_file = os.path.join(
        OUTPUT_DIRECTORY,
        "generalization_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    print()
    print(
        "Generalization results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Comparison results saved to:"
    )

    print(
        comparison_file
    )

    print()
    print(
        "GENERALIZATION TEST COMPLETED!"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()