from env import DEVICE
import torch.nn as nn


def build_mlp(input_dim:int, h_dims:list[int]):
    layers = []
    for h_dim in h_dims:
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.Tanh())
        input_dim = h_dim
    return nn.Sequential(*layers)

def build_embedding(input_dim:int, h_dim:int):
    return nn.Embedding(input_dim, h_dim)

def gen_policy(policy_args:list[dict]):
    models = []
    for policy_arg in policy_args:
        model = eval(f"build_{policy_arg['type']}")(**policy_arg['args'])
        models.append(model)
    return nn.Sequential(*models).to(DEVICE)