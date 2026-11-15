import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

DQN_TRAINING = (
    "results/csv/"
    "dqn_random_curriculum_training_results.csv"
)

DDQN_TRAINING = (
    "results/csv/"
    "ddqn_random_curriculum_training_results.csv"
)

DQN_TEST = (
    "results/csv/"
    "dqn_random_curriculum_test_results.csv"
)

DDQN_TEST = (
    "results/csv/"
    "ddqn_random_curriculum_test_results.csv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "results/comparison/random"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DQN_TRAINING,
    DDQN_TRAINING,
    DQN_TEST,
    DDQN_TEST
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            f"ERROR: File not found -> "
            f"{file_path}"
        )

        print(
            "Please make sure all training "
            "and testing files exist."
        )

        raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

dqn_train = pd.read_csv(
    DQN_TRAINING
)

ddqn_train = pd.read_csv(
    DDQN_TRAINING
)

dqn_test = pd.read_csv(
    DQN_TEST
)

ddqn_test = pd.read_csv(
    DDQN_TEST
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_column(df, *names):

    for name in names:

        if name in df.columns:

            return name

    return None


# ============================================================
# TRAINING METRICS
# ============================================================

dqn_train_reward_col = get_column(
    dqn_train,
    "Cumulative_Reward"
)

ddqn_train_reward_col = get_column(
    ddqn_train,
    "Cumulative_Reward"
)

dqn_train_loss_col = get_column(
    dqn_train,
    "Training_Loss"
)

ddqn_train_loss_col = get_column(
    ddqn_train,
    "Training_Loss"
)

dqn_train_success_col = get_column(
    dqn_train,
    "Success"
)

ddqn_train_success_col = get_column(
    ddqn_train,
    "Success"
)


# ============================================================
# TEST METRICS
# ============================================================

dqn_test_reward_col = get_column(
    dqn_test,
    "Cumulative_Reward"
)

ddqn_test_reward_col = get_column(
    ddqn_test,
    "Cumulative_Reward"
)

dqn_test_success_col = get_column(
    dqn_test,
    "Success"
)

ddqn_test_success_col = get_column(
    ddqn_test,
    "Success"
)

dqn_test_collision_col = get_column(
    dqn_test,
    "Collision"
)

ddqn_test_collision_col = get_column(
    ddqn_test,
    "Collision"
)

dqn_test_steps_col = get_column(
    dqn_test,
    "Episode_Length"
)

ddqn_test_steps_col = get_column(
    ddqn_test,
    "Episode_Length"
)

dqn_test_path_col = get_column(
    dqn_test,
    "Path_Length"
)

ddqn_test_path_col = get_column(
    ddqn_test,
    "Path_Length"
)

dqn_test_distance_col = get_column(
    dqn_test,
    "Final_Distance"
)

ddqn_test_distance_col = get_column(
    ddqn_test,
    "Final_Distance"
)


# ============================================================
# CALCULATE TRAINING STATISTICS
# ============================================================

dqn_avg_training_reward = (
    dqn_train[
        dqn_train_reward_col
    ].mean()
)

ddqn_avg_training_reward = (
    ddqn_train[
        ddqn_train_reward_col
    ].mean()
)


dqn_avg_training_loss = (
    dqn_train[
        dqn_train_loss_col
    ].mean()
)

ddqn_avg_training_loss = (
    ddqn_train[
        ddqn_train_loss_col
    ].mean()
)


dqn_training_success_rate = (
    dqn_train[
        dqn_train_success_col
    ].mean()
    * 100
)

ddqn_training_success_rate = (
    ddqn_train[
        ddqn_train_success_col
    ].mean()
    * 100
)


# ============================================================
# FINAL TRAINING PERFORMANCE
# ============================================================

dqn_final_reward = (
    dqn_train[
        dqn_train_reward_col
    ].tail(20).mean()
)

ddqn_final_reward = (
    ddqn_train[
        ddqn_train_reward_col
    ].tail(20).mean()
)


# ============================================================
# TEST STATISTICS
# ============================================================

dqn_test_success_rate = (
    dqn_test[
        dqn_test_success_col
    ].mean()
    * 100
)

ddqn_test_success_rate = (
    ddqn_test[
        ddqn_test_success_col
    ].mean()
    * 100
)


dqn_test_collision_rate = (
    dqn_test[
        dqn_test_collision_col
    ].mean()
    * 100
)

ddqn_test_collision_rate = (
    ddqn_test[
        ddqn_test_collision_col
    ].mean()
    * 100
)


dqn_avg_test_reward = (
    dqn_test[
        dqn_test_reward_col
    ].mean()
)

ddqn_avg_test_reward = (
    ddqn_test[
        ddqn_test_reward_col
    ].mean()
)


dqn_avg_test_steps = (
    dqn_test[
        dqn_test_steps_col
    ].mean()
)

ddqn_avg_test_steps = (
    ddqn_test[
        ddqn_test_steps_col
    ].mean()
)


dqn_avg_test_path = (
    dqn_test[
        dqn_test_path_col
    ].mean()
)

ddqn_avg_test_path = (
    ddqn_test[
        ddqn_test_path_col
    ].mean()
)


dqn_avg_distance = (
    dqn_test[
        dqn_test_distance_col
    ].mean()
)

ddqn_avg_distance = (
    ddqn_test[
        ddqn_test_distance_col
    ].mean()
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Metric": [

        "Average Training Reward",

        "Average Training Loss",

        "Training Success Rate (%)",

        "Final 20-Episode Reward",

        "Test Success Rate (%)",

        "Test Collision Rate (%)",

        "Average Test Reward",

        "Average Test Episode Length",

        "Average Test Path Length",

        "Average Final Distance"
    ],

    "DQN": [

        dqn_avg_training_reward,

        dqn_avg_training_loss,

        dqn_training_success_rate,

        dqn_final_reward,

        dqn_test_success_rate,

        dqn_test_collision_rate,

        dqn_avg_test_reward,

        dqn_avg_test_steps,

        dqn_avg_test_path,

        dqn_avg_distance
    ],

    "DDQN": [

        ddqn_avg_training_reward,

        ddqn_avg_training_loss,

        ddqn_training_success_rate,

        ddqn_final_reward,

        ddqn_test_success_rate,

        ddqn_test_collision_rate,

        ddqn_avg_test_reward,

        ddqn_avg_test_steps,

        ddqn_avg_test_path,

        ddqn_avg_distance
    ]
})


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_path = (
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print()

print("=" * 75)

print(
    "RANDOM DQN VS RANDOM DDQN COMPARISON"
)

print("=" * 75)

print(
    comparison.to_string(
        index=False
    )
)

print("=" * 75)


# ============================================================
# GRAPH 1
# TRAINING REWARD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    dqn_train["Episode"],
    dqn_train[
        dqn_train_reward_col
    ],
    label="DQN",
    alpha=0.5
)

plt.plot(
    ddqn_train["Episode"],
    ddqn_train[
        ddqn_train_reward_col
    ],
    label="DDQN",
    alpha=0.5
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Cumulative Reward"
)

plt.title(
    "DQN vs DDQN - Random Environment Training Reward"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_training_reward.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2
# TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    dqn_train["Episode"],
    dqn_train[
        dqn_train_loss_col
    ],
    label="DQN",
    alpha=0.7
)

plt.plot(
    ddqn_train["Episode"],
    ddqn_train[
        ddqn_train_loss_col
    ],
    label="DDQN",
    alpha=0.7
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Training Loss"
)

plt.title(
    "DQN vs DDQN - Training Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_training_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3
# TRAINING SUCCESS RATE
# ============================================================

dqn_train_success = (
    dqn_train[
        dqn_train_success_col
    ]
    .rolling(
        window=20,
        min_periods=1
    )
    .mean()
    * 100
)

ddqn_train_success = (
    ddqn_train[
        ddqn_train_success_col
    ]
    .rolling(
        window=20,
        min_periods=1
    )
    .mean()
    * 100
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    dqn_train["Episode"],
    dqn_train_success,
    label="DQN"
)

plt.plot(
    ddqn_train["Episode"],
    ddqn_train_success,
    label="DDQN"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Success Rate (%)"
)

plt.title(
    "DQN vs DDQN - Moving Success Rate"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_success_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4
# TEST PERFORMANCE
# ============================================================

models = [
    "DQN",
    "DDQN"
]

success_values = [
    dqn_test_success_rate,
    ddqn_test_success_rate
]

collision_values = [
    dqn_test_collision_rate,
    ddqn_test_collision_rate
]


x = range(
    len(models)
)

width = 0.35


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    [
        i - width / 2
        for i in x
    ],
    success_values,
    width=width,
    label="Success Rate"
)

plt.bar(
    [
        i + width / 2
        for i in x
    ],
    collision_values,
    width=width,
    label="Collision Rate"
)

plt.xticks(
    list(x),
    models
)

plt.ylabel(
    "Percentage (%)"
)

plt.title(
    "DQN vs DDQN - Unseen Random Map Performance"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_test_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 5
# TEST REWARD
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    models,
    [
        dqn_avg_test_reward,
        ddqn_avg_test_reward
    ]
)

plt.ylabel(
    "Average Cumulative Reward"
)

plt.title(
    "DQN vs DDQN - Average Test Reward"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_test_reward.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 6
# FINAL DISTANCE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    models,
    [
        dqn_avg_distance,
        ddqn_avg_distance
    ]
)

plt.ylabel(
    "Average Final Distance"
)

plt.title(
    "DQN vs DDQN - Final Distance on Unseen Maps"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    +
    "/random_dqn_vs_ddqn_final_distance.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print("=" * 75)

print(
    "COMPARISON COMPLETED SUCCESSFULLY"
)

print("=" * 75)

print(
    f"Comparison CSV:"
)

print(
    comparison_path
)

print()

print(
    "Graphs saved in:"
)

print(
    OUTPUT_DIR
)

print("=" * 75)