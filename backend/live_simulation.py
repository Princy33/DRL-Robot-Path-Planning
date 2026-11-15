import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add backend directory to Python path
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 20
MAX_STEPS = 150

MODEL_PATH = os.path.join(
    "models",
    "dqn_model.pth"
)

OUTPUT_DIR = os.path.join(
    "results",
    "paths"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "dqn_live_simulation.png"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS
)


# ============================================================
# CREATE DQN AGENT
# ============================================================

agent = DQNAgent(
    state_size=405,
    action_size=4
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print(
        f"ERROR: Model not found: {MODEL_PATH}"
    )

    raise SystemExit


agent.load(
    MODEL_PATH
)

agent.epsilon = 0.0

print(
    "Trained DQN model loaded successfully."
)


# ============================================================
# RESET ENVIRONMENT
# ============================================================

state, info = env.reset(
    seed=42
)

path = [
    tuple(env.robot_position)
]

total_reward = 0.0
success = False
collision = False


# ============================================================
# RUN GREEDY SIMULATION
# ============================================================

for step in range(MAX_STEPS):

    action = agent.select_action(
        state,
        training=False
    )

    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    path.append(
        tuple(env.robot_position)
    )

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


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 60)
print("LIVE DQN ROBOT SIMULATION")
print("=" * 60)

print(
    f"Steps          : {step + 1}"
)

print(
    f"Path Length    : {len(path)}"
)

print(
    f"Total Reward   : {total_reward:.2f}"
)

print(
    f"Success        : {success}"
)

print(
    f"Collision      : {collision}"
)

print(
    f"Final Position : {env.robot_position}"
)

print(
    f"Goal Position  : {env.goal_position}"
)

print("=" * 60)


# ============================================================
# PREPARE GRID
# ============================================================

grid = np.zeros(
    (GRID_SIZE, GRID_SIZE)
)


# Obstacles
for obstacle in env.obstacles:

    row, col = obstacle

    grid[row, col] = 1


# ============================================================
# CREATE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 10)
)

plt.imshow(
    grid,
    cmap="Greys",
    origin="upper"
)


# ============================================================
# DRAW PATH
# ============================================================

path_rows = [
    position[0]
    for position in path
]

path_cols = [
    position[1]
    for position in path
]

plt.plot(
    path_cols,
    path_rows,
    linewidth=3,
    marker="o",
    markersize=3,
    label="Robot Path"
)


# ============================================================
# START POSITION
# ============================================================

start_row, start_col = path[0]

plt.scatter(
    start_col,
    start_row,
    s=180,
    marker="s",
    label="Start"
)


# ============================================================
# GOAL POSITION
# ============================================================

goal_row, goal_col = env.goal_position

plt.scatter(
    goal_col,
    goal_row,
    s=220,
    marker="*",
    label="Goal"
)


# ============================================================
# FINAL ROBOT POSITION
# ============================================================

final_row, final_col = env.robot_position

plt.scatter(
    final_col,
    final_row,
    s=120,
    marker="o",
    label="Final Position"
)


# ============================================================
# GRID SETTINGS
# ============================================================

plt.xticks(
    range(GRID_SIZE)
)

plt.yticks(
    range(GRID_SIZE)
)

plt.grid(
    True,
    alpha=0.3
)

plt.xlabel(
    "Grid Column"
)

plt.ylabel(
    "Grid Row"
)

plt.title(
    "DQN Robot Path Planning Simulation"
)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print(
    f"Simulation graph saved to:"
)

print(
    OUTPUT_PATH
)