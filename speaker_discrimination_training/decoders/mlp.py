"""MLP Decoder definitions.
"""

from torch import nn

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, use_bn=True):
        super().__init__()

        self.lin1 = nn.Linear(dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, projection_size)
        self.lin3 = nn.Linear(projection_size, 1)

        self.use_bn = use_bn

        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()

        '''self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )'''

    def forward(self, x):
        x = self.lin1(x)
        # if self.use_bn:
        #     x = self.bn(x)
        x = self.relu(x)
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        return x


# import torch
# model = MLP(1280, 256, 4096)
# x = torch.randn(128, 1, 1280)

# # Let's print it
# model(x)