"""MLP Decoder definitions.
"""

from torch import nn

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, use_bn=True):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    import torch
    model = MLP(1280, 256, 4096)
    x = torch.randn(128, 1, 1280)
    # Let's print it
    model(x)