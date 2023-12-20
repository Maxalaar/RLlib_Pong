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

        dummy_input = torch.randn((10, *self.obs_shape)).permute(0, 3, 1, 2)

        # Paramètres personnalisables
        self.conv_layers = model_config.get("conv_layers", [64, 32])  # Couches convolutives
        self.kernel_sizes = model_config.get("kernel_sizes", [(4, 4), ])  # Tailles des noyaux
        self.fc_layers = model_config.get("fc_layers", [128, 128])         # Couches fully connected
        self.activation = model_config.get("activation", "relu")      # Fonction d'activation

        # # Couches convolutives
        # conv_layers = []
        # in_channels = self.obs_shape[0]
        # for out_channels, kernel_size in zip(self.conv_layers, self.kernel_sizes):
        #     conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
        #     conv_layers.append(get_activation(self.activation))
        #     in_channels = out_channels
        #
        # self.conv = nn.Sequential(*conv_layers)

        self.conv = nn.Sequential(*[
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
        ])

        # Couches fully connected
        fc_layers = []
        in_features = self.conv(dummy_input).flatten(start_dim=1).shape[1]
        for units in self.fc_layers:
            fc_layers.append(nn.Linear(in_features, units))
            fc_layers.append(get_activation(self.activation))
            in_features = units

        self.fc = nn.Sequential(*fc_layers)

        # Couche de sortie
        self.output_layer = nn.Linear(self.fc_layers[-1], num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        output = self.output_layer(x)
        return output, state
