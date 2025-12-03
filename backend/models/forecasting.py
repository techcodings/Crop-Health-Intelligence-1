"""
More advanced temporal forecaster (LSTM stack).
"""

import torch
import torch.nn as nn

class SimpleTemporalForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, horizon: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)
