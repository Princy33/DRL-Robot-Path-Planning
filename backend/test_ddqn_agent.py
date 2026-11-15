import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from backend.ddqn_agent import DDQNAgent


STATE_SIZE = 405
ACTION_SIZE = 4


agent = DDQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE
)

print("=" * 70)
print("DDQN AGENT TEST")
print("=" * 70)

print("Device       :", agent.device)
print("State Size   :", agent.state_size)
print("Action Size  :", agent.action_size)
print("Epsilon      :", agent.epsilon)
print("Memory Size  :", len(agent.memory))

print()

print("Testing random state action selection...")

import numpy as np

state = np.zeros(STATE_SIZE, dtype=np.float32)

action = agent.select_action(
    state,
    training=False
)

print("Selected Action:", action)

print()

print("Testing target network update...")

agent.update_target_network()

print("Target network updated successfully.")

print()
print("=" * 70)
print("DDQN AGENT TEST PASSED")
print("=" * 70)