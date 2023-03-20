"""MLP Decoder definitions.
"""

from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size, layers_sizes):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for size in layers_sizes:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Linear(size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



if __name__ == '__main__':
    import torch
    input_size = 1280
    layers_sizes = [4096, 256, 256, 128, 64]
    model = MLP(input_size, layers_sizes)
    x = torch.randn(128, 1, 1280)
    # Let's print it
    output = model(x)
    print(output.shape)