import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

CSV_PATH = "results/csv/dqn_training_results.csv"
GRAPH_DIR = "results/graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)


# ---------------------------------------------------------
# LOAD TRAINING RESULTS
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("DQN TRAINING RESULTS VISUALIZATION")
print("=" * 70)

print("\nCSV loaded successfully.")
print(f"Number of Episodes: {len(df)}")
print("\nAvailable Columns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 1. EPISODE VS CUMULATIVE REWARD
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Cumulative_Reward"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Episode vs Cumulative Reward")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "episode_vs_cumulative_reward.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 2. EPISODE VS AVERAGE REWARD
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Average_Reward"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Episode vs Average Reward")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "episode_vs_average_reward.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 3. EPISODE VS TRAINING LOSS
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Training_Loss"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Training Loss")
plt.title("Episode vs Training Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "episode_vs_training_loss.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 4. EPISODE VS EPISODE LENGTH
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Episode_Length"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Episode Length (Steps)")
plt.title("Episode vs Episode Length")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "episode_vs_episode_length.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 5. EXPLORATION RATE VS EPISODE
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Epsilon"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Exploration Rate (Epsilon)")
plt.title("Exploration Rate vs Episode")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "exploration_rate_vs_episode.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# COMPLETION MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ALL TRAINING GRAPHS GENERATED SUCCESSFULLY")
print("=" * 70)

print("\nGraphs saved in:")
print("results/graphs/")

print("\nGenerated files:")

for file in sorted(os.listdir(GRAPH_DIR)):
    if file.endswith(".png"):
        print(f"  ✓ {file}")

print("=" * 70)