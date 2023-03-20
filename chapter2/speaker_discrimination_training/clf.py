import torch.nn as nn

class Clf(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        return self.net(x)