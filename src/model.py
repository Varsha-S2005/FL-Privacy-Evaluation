
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_features=12):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
