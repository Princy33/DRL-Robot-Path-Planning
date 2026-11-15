import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DDQN TRAINING RESULTS VISUALIZATION
# ============================================================

CSV_PATH = "results/csv/ddqn_training_results.csv"
OUTPUT_DIR = "results/graphs/ddqn"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

print("CSV loaded successfully.")
print("Number of Episodes:", len(df))
print("Available Columns:")
print(df.columns.tolist())

# ============================================================
# 1. Episode vs Cumulative Reward
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Cumulative_Reward"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("DDQN: Episode vs Cumulative Reward")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ddqn_episode_vs_cumulative_reward.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 2. Episode vs Average Reward
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Average_Reward"],
    linewidth=2
)

plt.xlabel("Episode")
plt.ylabel("Moving Average Reward")
plt.title("DDQN: Episode vs Average Reward")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ddqn_episode_vs_average_reward.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 3. Episode vs Training Loss
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Training_Loss"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Training Loss")
plt.title("DDQN: Episode vs Training Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ddqn_episode_vs_training_loss.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 4. Episode vs Episode Length
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Episode_Length"],
    linewidth=1
)

plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("DDQN: Episode vs Episode Length")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ddqn_episode_vs_episode_length.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 5. Exploration Rate vs Episode
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df["Episode"],
    df["Epsilon"],
    linewidth=2
)

plt.xlabel("Episode")
plt.ylabel("Exploration Rate (Epsilon)")
plt.title("DDQN: Exploration Rate vs Episode")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ddqn_exploration_rate_vs_episode.png"
    ),
    dpi=300
)

plt.close()

print()
print("=" * 70)
print("ALL DDQN TRAINING GRAPHS GENERATED SUCCESSFULLY")
print("=" * 70)
print("Graphs saved in:")
print(OUTPUT_DIR)
print()
print("Generated:")
print("1. ddqn_episode_vs_cumulative_reward.png")
print("2. ddqn_episode_vs_average_reward.png")
print("3. ddqn_episode_vs_training_loss.png")
print("4. ddqn_episode_vs_episode_length.png")
print("5. ddqn_exploration_rate_vs_episode.png")