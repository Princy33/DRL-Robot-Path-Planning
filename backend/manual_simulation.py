import os
import sys
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from environment import RobotPathPlanningEnv


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 20
MAX_STEPS = 150


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = RobotPathPlanningEnv(
    grid_size=GRID_SIZE,
    max_steps=MAX_STEPS
)

state, info = env.reset(seed=42)

print()
print("=" * 60)
print("MANUAL ROBOT PATH PLANNING SIMULATOR")
print("=" * 60)

print("Start Position :", env.robot_position)
print("Goal Position  :", env.goal_position)

print()
print("Controls:")
print("W = Up")
print("S = Down")
print("A = Left")
print("D = Right")
print("Q = Quit")

print("=" * 60)


# ============================================================
# ACTION MAPPING
# ============================================================

action_map = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3
}


# ============================================================
# SIMULATION LOOP
# ============================================================

total_reward = 0.0

while True:

    print()
    print(
        f"Robot Position: "
        f"{env.robot_position}"
    )

    command = input(
        "Enter movement: "
    ).lower().strip()

    if command == "q":

        print(
            "Simulation stopped."
        )

        break

    if command not in action_map:

        print(
            "Invalid input. "
            "Use W, A, S, D or Q."
        )

        continue

    action = action_map[
        command
    ]

    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    print(
        f"Reward: {reward:.2f}"
    )

    if info.get(
        "collision",
        False
    ):

        print(
            "⚠️ Collision detected!"
        )

    if info.get(
        "success",
        False
    ):

        print()
        print(
            "🎯 GOAL REACHED!"
        )

        print(
            f"Total Reward: "
            f"{total_reward:.2f}"
        )

        break

    if terminated or truncated:

        print()
        print(
            "Episode finished."
        )

        print(
            f"Total Reward: "
            f"{total_reward:.2f}"
        )

        break


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 60)

print(
    "FINAL ROBOT POSITION:",
    env.robot_position
)

print(
    "GOAL POSITION:",
    env.goal_position
)

print(
    "TOTAL STEPS:",
    env.steps
)

print(
    "TOTAL REWARD:",
    round(
        total_reward,
        2
    )
)

print("=" * 60)