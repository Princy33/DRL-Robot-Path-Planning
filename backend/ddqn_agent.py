import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from neural_network import DQNNetwork
from replay_memory import ReplayMemory


class DDQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_capacity=100000,
        batch_size=64
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

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
            state_size,
            action_size
        ).to(self.device)

        # --------------------------------------------------
        # Target Network
        # --------------------------------------------------

        self.target_network = DQNNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # --------------------------------------------------
        # Freeze Target Network
        # --------------------------------------------------

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

    # ======================================================
    # ACTION SELECTION
    # ======================================================

    def select_action(self, state, training=True):

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:

            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        return int(torch.argmax(q_values, dim=1).item())

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
    # DDQN TRAINING STEP
    # ======================================================

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        states = torch.tensor(
            np.array(states),
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
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # --------------------------------------------------
        # Current Q-values
        # --------------------------------------------------

        current_q_values = self.policy_network(
            states
        ).gather(1, actions).squeeze(1)

        # --------------------------------------------------
        # DOUBLE DQN TARGET
        # --------------------------------------------------
        #
        # Step 1:
        # Policy network selects the best action.
        #
        # Step 2:
        # Target network evaluates that selected action.
        #
        # This is the main difference from standard DQN.
        # --------------------------------------------------

        with torch.no_grad():

            # Policy network chooses action
            next_actions = self.policy_network(
                next_states
            ).argmax(dim=1, keepdim=True)

            # Target network evaluates that action
            next_q_values = self.target_network(
                next_states
            ).gather(
                1,
                next_actions
            ).squeeze(1)

            target_q_values = rewards + (
                self.gamma *
                next_q_values *
                (1 - dones)
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return float(loss.item())

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

    def save(self, path):

        torch.save(
            self.policy_network.state_dict(),
            path
        )

        print(
            f"DDQN model saved successfully: {path}"
        )

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def load(self, path):

        state_dict = torch.load(
            path,
            map_location=self.device
        )

        self.policy_network.load_state_dict(
            state_dict
        )

        self.target_network.load_state_dict(
            state_dict
        )

        print(
            f"DDQN model loaded successfully: {path}"
        )