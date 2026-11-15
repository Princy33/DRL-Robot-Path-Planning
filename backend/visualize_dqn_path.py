import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

GRID_SIZE = 20
MAX_STEPS = 150
MODEL_PATH = "models/dqn_model.pth"

OUTPUT_PATH = "results/paths/dqn_greedy_path.png"

os.makedirs("results/paths", exist_ok=True)


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS
)


# ---------------------------------------------------------
# AGENT
# ---------------------------------------------------------

agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n
)

agent.load(MODEL_PATH)
agent.epsilon = 0.0


# ---------------------------------------------------------
# RUN TRAINED POLICY
# ---------------------------------------------------------

state, info = env.reset(seed=42)

done = False

while not done:

    state_tensor = torch.tensor(
        np.asarray(state, dtype=np.float32),
        dtype=torch.float32
    ).unsqueeze(0).to(agent.device)

    with torch.no_grad():
        q_values = agent.policy_network(state_tensor)

    action = int(torch.argmax(q_values, dim=1).item())

    next_state, reward, terminated, truncated, info = env.step(action)

    state = next_state

    done = terminated or truncated


# ---------------------------------------------------------
# CREATE GRID
# ---------------------------------------------------------

grid = np.zeros((GRID_SIZE, GRID_SIZE))

for obstacle in env.obstacles:
    row, col = obstacle
    grid[row, col] = 1


# ---------------------------------------------------------
# PLOT ENVIRONMENT
# ---------------------------------------------------------

plt.figure(figsize=(10, 10))

plt.imshow(
    grid,
    cmap="Greys",
    origin="upper"
)

# ---------------------------------------------------------
# PLOT PATH
# ---------------------------------------------------------

path_rows = [position[0] for position in env.path]
path_cols = [position[1] for position in env.path]

plt.plot(
    path_cols,
    path_rows,
    linewidth=2,
    marker="o",
    markersize=3,
    label="DQN Path"
)


# ---------------------------------------------------------
# START AND GOAL
# ---------------------------------------------------------

start_row, start_col = env.start_position
goal_row, goal_col = env.goal_position

plt.scatter(
    start_col,
    start_row,
    s=180,
    marker="s",
    label="Start"
)

plt.scatter(
    goal_col,
    goal_row,
    s=220,
    marker="*",
    label="Goal"
)


# ---------------------------------------------------------
# LABELS
# ---------------------------------------------------------

plt.title(
    "DQN-Based Robot Path Planning",
    fontsize=16
)

plt.xlabel("Grid Column")
plt.ylabel("Grid Row")

plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("=" * 70)
print("DQN PATH VISUALIZATION")
print("=" * 70)

print(f"Final Position : {env.robot_position}")
print(f"Goal Position  : {env.goal_position}")
print(f"Steps Taken    : {env.steps}")
print(f"Path Length    : {len(env.path)}")
print(f"Success        : {info.get('success', False)}")

print("\nPath visualization saved to:")
print(OUTPUT_PATH)

print("=" * 70)