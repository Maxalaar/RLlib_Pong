import tensorflow as tf
import torch
import torch.nn as nn

if __name__ == '__main__':
    conv = nn.Conv2d(4, 10, 3, stride=2)
    data = torch.randn((100, 84, 84, 4))

    print(conv(data).shape)
