import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # 🔑 LazyLinear automatically infers input size
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
