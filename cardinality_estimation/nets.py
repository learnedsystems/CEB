import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleRegression(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size):
        super(SimpleRegression, self).__init__()

        self.layers = nn.ModuleList()
        layer1 = nn.Sequential(
            nn.Linear(input_width, hidden_layer_size, bias=True),
            nn.ReLU()
        ).to(device)

        self.layers.append(layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=True),
                nn.ReLU()
            ).to(device)
            self.layers.append(layer)

        final_layer = nn.Sequential(
            nn.Linear(hidden_layer_size, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.layers.append(final_layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
