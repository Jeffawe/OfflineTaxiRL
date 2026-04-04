import torch
import torch.nn as nn

class BehaviorCloningModel(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = 4) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)