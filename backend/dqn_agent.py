import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from neural_network import DQNNetwork
from replay_memory import ReplayMemory


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Robot Path Planning.

    Implements:
    - Epsilon-greedy exploration
    - Experience replay
    - Policy network
    - Target network
    - Bellman Q-learning update
    - Huber loss
    - Adam optimizer
    """

    def __init__(
        self,
        state_size=405,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_capacity=100_000,
        batch_size=64,
        target_update_frequency=10
    ):

        # --------------------------------------------------
        # Basic Configuration
        # --------------------------------------------------

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Training parameters
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --------------------------------------------------
        # Policy Network
        # --------------------------------------------------

        self.policy_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size
        ).to(self.device)

        # --------------------------------------------------
        # Target Network
        # --------------------------------------------------

        self.target_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size
        ).to(self.device)

        # Initially both networks have identical weights
        self.update_target_network()

        # Target network does not require gradients
        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        # --------------------------------------------------
        # Loss Function
        # --------------------------------------------------

        self.loss_function = nn.SmoothL1Loss()

        # --------------------------------------------------
        # Replay Memory
        # --------------------------------------------------

        self.memory = ReplayMemory(
            capacity=replay_capacity
        )

        # --------------------------------------------------
        # Training Counters
        # --------------------------------------------------

        self.training_steps = 0

    # ======================================================
    # ACTION SELECTION
    # ======================================================

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy strategy.
        """

        # Exploration
        if training and random.random() < self.epsilon:

            return random.randrange(self.action_size)

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1
        ).item()

        return action

    # ======================================================
    # STORE EXPERIENCE
    # ======================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ======================================================
    # TRAIN FROM REPLAY MEMORY
    # ======================================================

    def train_step(self):

        # Not enough experiences
        if not self.memory.can_sample(
            self.batch_size
        ):

            return None

        # --------------------------------------------------
        # Sample Mini-Batch
        # --------------------------------------------------

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )

        # Convert to PyTorch tensors
        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        # --------------------------------------------------
        # Current Q Values
        # --------------------------------------------------

        current_q_values = self.policy_network(
            states
        ).gather(
            1,
            actions
        )

        # --------------------------------------------------
        # Target Q Values
        # --------------------------------------------------

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(
                dim=1,
                keepdim=True
            )[0]

            target_q_values = rewards + (
                self.gamma
                * next_q_values
                * (1 - dones)
            )

        # --------------------------------------------------
        # Calculate Loss
        # --------------------------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # Gradient clipping improves training stability
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.training_steps += 1

        return loss.item()

    # ======================================================
    # UPDATE TARGET NETWORK
    # ======================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    # ======================================================
    # EPSILON DECAY
    # ======================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    def save(self, filepath):

        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps
            },
            filepath
        )

        print(f"Model saved successfully: {filepath}")

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def load(self, filepath):

        checkpoint = torch.load(
            filepath,
            map_location=self.device
        )

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon_min
        )

        self.training_steps = checkpoint.get(
            "training_steps",
            0
        )

        print(f"Model loaded successfully: {filepath}")


# ==========================================================
# TEST DQN AGENT
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DQN AGENT TEST")
    print("=" * 60)

    # Create agent
    agent = DQNAgent(
        state_size=405,
        action_size=4
    )

    print("\nDevice:")
    print(agent.device)

    print("\nState Size:")
    print(agent.state_size)

    print("\nAction Size:")
    print(agent.action_size)

    print("\nInitial Epsilon:")
    print(agent.epsilon)

    print("\nReplay Memory:")
    print(len(agent.memory))

    # Create dummy state
    sample_state = np.random.rand(
        405
    ).astype(np.float32)

    # Select action
    action = agent.select_action(
        sample_state,
        training=True
    )

    print("\nSelected Action:")
    print(action)

    print("\nTarget Network:")
    print("Successfully initialized")

    print("\nOptimizer:")
    print(agent.optimizer.__class__.__name__)

    print("\nLoss Function:")
    print(agent.loss_function.__class__.__name__)

    print("\nDQN Agent is working successfully!")