import torch
import torch.nn as nn

class SimpleConv(nn.Module):
        def __init__(self, input_size, hidden_size, n_classes):
            super(SimpleConv, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            out = self.linear1(x)
            out = self.relu(out)
            out = self.linear2(out)
            return out