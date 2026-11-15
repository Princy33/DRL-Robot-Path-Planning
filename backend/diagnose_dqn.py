import os
import numpy as np

from environment import RobotPathPlanningEnv
from dqn_agent import DQNAgent


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

GRID_SIZE = 20
MAX_STEPS = 150
MODEL_PATH = "models/dqn_model.pth"

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS
)

agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n
)

agent.load(MODEL_PATH)
agent.epsilon = 0.0


# ---------------------------------------------------------
# ACTION NAMES
# ---------------------------------------------------------

action_names = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


# ---------------------------------------------------------
# RESET
# ---------------------------------------------------------

state, info = env.reset(seed=42)

print("=" * 70)
print("DQN POLICY DIAGNOSIS")
print("=" * 70)

print(f"\nStart Position : {env.robot_position}")
print(f"Goal Position  : {env.goal_position}")
print(f"State Size     : {len(state)}")

print("\nFirst 30 Greedy Actions:")
print("-" * 70)


# ---------------------------------------------------------
# RUN GREEDY POLICY
# ---------------------------------------------------------

positions = []

for step in range(1, 31):

    # Get Q-values directly
    state_tensor = np.asarray(state, dtype=np.float32)

    import torch

    state_tensor = torch.tensor(
        state_tensor,
        dtype=torch.float32
    ).unsqueeze(0).to(agent.device)

    with torch.no_grad():
        q_values = agent.policy_network(state_tensor)

    q_values = q_values.cpu().numpy()[0]

    action = int(np.argmax(q_values))

    old_position = env.robot_position

    next_state, reward, terminated, truncated, info = env.step(action)

    new_position = env.robot_position

    positions.append(new_position)

    print(
        f"Step {step:02d} | "
        f"Position {old_position} -> {new_position} | "
        f"Action: {action_names[action]:5s} | "
        f"Q-values: {np.round(q_values, 2)} | "
        f"Reward: {reward:7.2f}"
    )

    state = next_state

    if terminated or truncated:
        print("\nEpisode ended.")
        break


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

print(f"Final Position : {env.robot_position}")
print(f"Goal Position  : {env.goal_position}")
print(f"Steps Taken    : {env.steps}")
print(f"Path Length    : {len(env.path)}")

print("\nPath:")
print(env.path[:30])

print("=" * 70)