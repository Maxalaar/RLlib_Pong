import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activation function {activation} not supported.")


class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_shape = obs_space.shape
        self.num_outputs = num_outputs

        dummy_input = torch.randn(self.obs_shape)

        # Paramètres personnalisables
        self.conv_layers = model_config.get("conv_layers", [64, 32])  # Couches convolutives
        self.kernel_sizes = model_config.get("kernel_sizes", [(3, 3), (2, 2)])  # Tailles des noyaux
        self.fc_layers = model_config.get("fc_layers", [128])         # Couches fully connected
        self.activation = model_config.get("activation", "relu")      # Fonction d'activation

        # Couches convolutives
        conv_layers = []
        in_channels = self.obs_shape[0]
        for out_channels, kernel_size in zip(self.conv_layers, self.kernel_sizes):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            conv_layers.append(get_activation(self.activation))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Couches fully connected
        fc_layers = []
        in_features = np.prod(self.conv(dummy_input).shape)
        for units in self.fc_layers:
            fc_layers.append(nn.Linear(in_features, units))
            fc_layers.append(get_activation(self.activation))
            in_features = units

        self.fc = nn.Sequential(*fc_layers)

        # Couche de sortie
        self.output_layer = nn.Linear(self.fc_layers[-1], num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()   # .float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.output_layer(x)
        return output, state
