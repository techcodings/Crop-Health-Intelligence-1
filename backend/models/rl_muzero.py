import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleIrrigationPolicy(nn.Module):
    """
    Advanced placeholder for MuZero-style policy/value head.
    State is already a rich latent from the encoder.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_actions: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.gelu(self.fc1(state))
        x = F.gelu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
