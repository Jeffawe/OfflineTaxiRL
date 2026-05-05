import torch
import torch.nn as nn

# This can either be 128 or 256
num_features = 256

class BehaviorCloningModel(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = 4) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)