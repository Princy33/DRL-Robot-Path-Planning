import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Robot Path Planning.

    Input:
        405-dimensional environment state

    Output:
        Q-values for 4 possible actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right
    """

    def __init__(self, state_size=405, action_size=4):

        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(

            # Input Layer -> Hidden Layer 1
            nn.Linear(state_size, 256),
            nn.ReLU(),

            # Hidden Layer 1 -> Hidden Layer 2
            nn.Linear(256, 128),
            nn.ReLU(),

            # Hidden Layer 2 -> Hidden Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output Layer
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        """
        Perform forward pass through the network.
        """

        return self.network(state)


# ==========================================================
# TEST THE NETWORK
# ==========================================================

if __name__ == "__main__":

    # Environment state size
    state_size = 405

    # Number of possible actions
    action_size = 4

    # Create DQN model
    model = DQNNetwork(
        state_size=state_size,
        action_size=action_size
    )

    print("=" * 60)
    print("DQN NEURAL NETWORK TEST")
    print("=" * 60)

    print("\nNetwork Architecture:")
    print(model)

    # Create a dummy state
    sample_state = torch.rand(1, state_size)

    print("\nInput State Shape:")
    print(sample_state.shape)

    # Pass state through network
    q_values = model(sample_state)

    print("\nOutput Q-Values:")
    print(q_values)

    print("\nOutput Shape:")
    print(q_values.shape)

    print("\nNumber of Trainable Parameters:")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(total_parameters)

    print("\nDQN Neural Network is working successfully!")