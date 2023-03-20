"""Aggregation Methods Definitions
"""

import torch
import torch.nn as nn

def add(x, y):
    return torch.add(x, y)
def concat(x,y):
    return torch.cat((x, y), 2)

class Aggregator(nn.Module):
    def __init__(self, aggregation):
        super().__init__()
        self.aggregator = self.agg_method(aggregation)
    
    def agg_method(self, method):
        if method == 'sum':
            return add
        if method == 'concat':
            return concat
    
    def forward(self, x, y):
        return self.aggregator(x, y)
    