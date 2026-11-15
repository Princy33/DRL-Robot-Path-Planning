import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DQN vs DDQN MODEL COMPARISON
# ============================================================

DQN_TRAINING_CSV = "results/csv/dqn_training_results.csv"
DDQN_TRAINING_CSV = "results/csv/ddqn_training_results.csv"

DQN_TEST_CSV = "results/csv/dqn_test_results.csv"
DDQN_TEST_CSV = "results/csv/ddqn_test_results.csv"

OUTPUT_DIR = "results/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

dqn_train = pd.read_csv(DQN_TRAINING_CSV)
ddqn_train = pd.read_csv(DDQN_TRAINING_CSV)

dqn_test = pd.read_csv(DQN_TEST_CSV)
ddqn_test = pd.read_csv(DDQN_TEST_CSV)

print("=" * 75)
print("DQN vs DDQN MODEL COMPARISON")
print("=" * 75)

# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("_", " ", regex=False)
    )

    return df


dqn_train = normalize_columns(dqn_train)
ddqn_train = normalize_columns(ddqn_train)

dqn_test = normalize_columns(dqn_test)
ddqn_test = normalize_columns(ddqn_test)

print("\nDQN Test Columns:")
print(dqn_test.columns.tolist())

print("\nDDQN Test Columns:")
print(ddqn_test.columns.tolist())

# ============================================================
# TRAINING METRICS
# ============================================================

dqn_training_time = dqn_train["Episode Time"].sum()
ddqn_training_time = ddqn_train["Episode Time"].sum()

dqn_avg_training_reward = dqn_train["Cumulative Reward"].mean()
ddqn_avg_training_reward = ddqn_train["Cumulative Reward"].mean()

dqn_avg_loss = dqn_train["Training Loss"].mean()
ddqn_avg_loss = ddqn_train["Training Loss"].mean()

dqn_training_success = dqn_train["Success"].mean() * 100
ddqn_training_success = ddqn_train["Success"].mean() * 100

# ============================================================
# FINAL MOVING AVERAGE REWARD
# ============================================================

if "Moving Average Reward" in dqn_train.columns:
    dqn_final_reward = dqn_train["Moving Average Reward"].iloc[-1]
else:
    dqn_final_reward = dqn_train["Average Reward"].iloc[-1]

if "Moving Average Reward" in ddqn_train.columns:
    ddqn_final_reward = ddqn_train["Moving Average Reward"].iloc[-1]
else:
    ddqn_final_reward = ddqn_train["Average Reward"].iloc[-1]

# ============================================================
# TEST METRICS
# ============================================================

dqn_test_success = dqn_test["Success"].mean() * 100
ddqn_test_success = ddqn_test["Success"].mean() * 100

dqn_test_collision = dqn_test["Collision"].mean() * 100
ddqn_test_collision = ddqn_test["Collision"].mean() * 100

dqn_test_reward = dqn_test["Cumulative Reward"].mean()
ddqn_test_reward = ddqn_test["Cumulative Reward"].mean()

dqn_test_steps = dqn_test["Episode Length"].mean()
ddqn_test_steps = ddqn_test["Episode Length"].mean()

dqn_test_path = dqn_test["Path Length"].mean()
ddqn_test_path = ddqn_test["Path Length"].mean()

# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Metric": [
        "Training Time (seconds)",
        "Average Training Reward",
        "Average Training Loss",
        "Training Success Rate (%)",
        "Final Moving Average Reward",
        "Test Success Rate (%)",
        "Test Collision Rate (%)",
        "Average Test Reward",
        "Average Test Episode Length",
        "Average Test Path Length"
    ],

    "DQN": [
        dqn_training_time,
        dqn_avg_training_reward,
        dqn_avg_loss,
        dqn_training_success,
        dqn_final_reward,
        dqn_test_success,
        dqn_test_collision,
        dqn_test_reward,
        dqn_test_steps,
        dqn_test_path
    ],

    "DDQN": [
        ddqn_training_time,
        ddqn_avg_training_reward,
        ddqn_avg_loss,
        ddqn_training_success,
        ddqn_final_reward,
        ddqn_test_success,
        ddqn_test_collision,
        ddqn_test_reward,
        ddqn_test_steps,
        ddqn_test_path
    ]
})

comparison["DQN"] = comparison["DQN"].round(2)
comparison["DDQN"] = comparison["DDQN"].round(2)

# ============================================================
# SAVE CSV
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "dqn_vs_ddqn_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

# ============================================================
# PRINT TABLE
# ============================================================

print("\n")
print("=" * 75)
print("FINAL COMPARISON TABLE")
print("=" * 75)

print(comparison.to_string(index=False))

# ============================================================
# GRAPH 1: TEST PERFORMANCE
# ============================================================

test_metrics = [
    "Success Rate (%)",
    "Average Test Reward",
    "Average Test Episode Length",
    "Average Test Path Length"
]

dqn_values = [
    dqn_test_success,
    dqn_test_reward,
    dqn_test_steps,
    dqn_test_path
]

ddqn_values = [
    ddqn_test_success,
    ddqn_test_reward,
    ddqn_test_steps,
    ddqn_test_path
]

x = range(len(test_metrics))
width = 0.35

plt.figure(figsize=(12, 7))

plt.bar(
    [i - width / 2 for i in x],
    dqn_values,
    width=width,
    label="DQN"
)

plt.bar(
    [i + width / 2 for i in x],
    ddqn_values,
    width=width,
    label="DDQN"
)

plt.xticks(
    list(x),
    test_metrics,
    rotation=15
)

plt.ylabel("Value")
plt.title("DQN vs DDQN Test Performance")
plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

test_graph = os.path.join(
    OUTPUT_DIR,
    "dqn_vs_ddqn_test_performance.png"
)

plt.savefig(
    test_graph,
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 2: TRAINING REWARD
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    dqn_train["Episode"],
    dqn_train["Cumulative Reward"],
    label="DQN",
    alpha=0.7
)

plt.plot(
    ddqn_train["Episode"],
    ddqn_train["Cumulative Reward"],
    label="DDQN",
    alpha=0.7
)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("DQN vs DDQN: Training Reward Comparison")
plt.legend()
plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

reward_graph = os.path.join(
    OUTPUT_DIR,
    "dqn_vs_ddqn_training_reward.png"
)

plt.savefig(
    reward_graph,
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 3: TRAINING LOSS
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    dqn_train["Episode"],
    dqn_train["Training Loss"],
    label="DQN",
    alpha=0.7
)

plt.plot(
    ddqn_train["Episode"],
    ddqn_train["Training Loss"],
    label="DDQN",
    alpha=0.7
)

plt.xlabel("Episode")
plt.ylabel("Training Loss")
plt.title("DQN vs DDQN: Training Loss Comparison")
plt.legend()
plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

loss_graph = os.path.join(
    OUTPUT_DIR,
    "dqn_vs_ddqn_training_loss.png"
)

plt.savefig(
    loss_graph,
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 4: SUCCESS RATE
# ============================================================

plt.figure(figsize=(8, 6))

models = [
    "DQN",
    "DDQN"
]

success_values = [
    dqn_test_success,
    ddqn_test_success
]

plt.bar(
    models,
    success_values
)

plt.ylabel("Success Rate (%)")
plt.ylim(0, 110)
plt.title("DQN vs DDQN Success Rate")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

success_graph = os.path.join(
    OUTPUT_DIR,
    "dqn_vs_ddqn_success_rate.png"
)

plt.savefig(
    success_graph,
    dpi=300
)

plt.close()

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles generated:")

print("1.", comparison_path)
print("2.", test_graph)
print("3.", reward_graph)
print("4.", loss_graph)
print("5.", success_graph)

print("\n")
print("=" * 75)