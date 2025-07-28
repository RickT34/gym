from env import *


def build_mlp(input_dim:int, h_dims):
    layers = []
    for h_dim in h_dims:
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.Tanh())
        input_dim = h_dim
    return nn.Sequential(*layers).to(DEVICE)