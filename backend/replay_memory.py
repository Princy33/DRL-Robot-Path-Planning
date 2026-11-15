import random
from collections import deque

import numpy as np


class ReplayMemory:
    """
    Experience Replay Memory for Deep Q-Network (DQN).

    Stores experiences in the form:

        (state, action, reward, next_state, done)

    and randomly samples mini-batches for training.
    """

    def __init__(self, capacity=100_000):

        self.capacity = capacity

        self.memory = deque(
            maxlen=capacity
        )

    # ==========================================================
    # STORE EXPERIENCE
    # ==========================================================

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.memory.append(experience)

    # ==========================================================
    # SAMPLE MINI-BATCH
    # ==========================================================

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    # ==========================================================
    # CURRENT MEMORY SIZE
    # ==========================================================

    def __len__(self):

        return len(self.memory)

    # ==========================================================
    # CHECK WHETHER ENOUGH EXPERIENCES EXIST
    # ==========================================================

    def can_sample(self, batch_size):

        return len(self.memory) >= batch_size


# ==========================================================
# TEST REPLAY MEMORY
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DQN EXPERIENCE REPLAY MEMORY TEST")
    print("=" * 60)

    # Create memory
    memory = ReplayMemory(
        capacity=1000
    )

    # Create dummy states
    state = np.zeros(405, dtype=np.float32)

    next_state = np.ones(
        405,
        dtype=np.float32
    )

    # Store sample experiences
    for i in range(20):

        action = i % 4
        reward = float(i)
        done = False

        memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    print("\nMemory Capacity :", memory.capacity)
    print("Stored Experiences :", len(memory))

    # Sample a mini-batch
    batch_size = 8

    if memory.can_sample(batch_size):

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = memory.sample(batch_size)

        print("\nSampled Mini-Batch:")
        print("States Shape      :", states.shape)
        print("Actions Shape     :", actions.shape)
        print("Rewards Shape     :", rewards.shape)
        print("Next States Shape :", next_states.shape)
        print("Dones Shape       :", dones.shape)

        print("\nReplay Memory is working successfully!")

    else:

        print("\nNot enough experiences for sampling.")