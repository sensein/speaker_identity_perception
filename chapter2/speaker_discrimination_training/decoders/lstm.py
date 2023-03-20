"""LSTM Decoder definitions.
"""

from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, 
                layers_hidden_sizes, 
                is_bidirectional=False,
                ):
        super(LSTM, self).__init__()

        self.lstm_layers = nn.ModuleList()
        for size in layers_hidden_sizes:
            self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=size, 
                                bidirectional=is_bidirectional, batch_first=True))
            input_size = size

        if is_bidirectional:
            size *= 2

        self.linear_layer = nn.Linear(size, 1)

    def forward(self, x):
        for layer in self.lstm_layers:
            x, (h,c) = layer(x)
        x = self.linear_layer(x[:,-1,:])
        return x



if __name__ == '__main__':
    import torch
    input_size = 10
    layers_sizes = [50]
    model = LSTM(input_size, layers_sizes)
    x = torch.randn(2, 5, 10)
    # Let's print it
    output = model(x)
    print(output.shape)