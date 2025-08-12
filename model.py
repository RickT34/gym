from env import DEVICE
import torch.nn as nn
import torch
import torch.distributions as D


def build_mlp(input_dim: int, h_dims: list[int]):
    layers = []
    for h_dim in h_dims:
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.Tanh())
        input_dim = h_dim
    return nn.Sequential(*layers)


def build_embedding(input_dim: int, h_dim: int):
    return nn.Embedding(input_dim, h_dim)


def gen_net(policy_args: list[dict]|dict):
    if isinstance(policy_args, dict):
        policy_args = [policy_args]
    models = []
    for policy_arg in policy_args:
        model = eval(f"build_{policy_arg['type']}")(**policy_arg["args"])
        models.append(model)
    return nn.Sequential(*models).to(DEVICE)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def get_dist(self, policy: torch.Tensor) -> D.Distribution:
        raise NotImplementedError


class CategoricalSampler(Sampler):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = torch.tensor(temperature, device=DEVICE)

    def get_dist(self, policy: torch.Tensor) -> D.Distribution:
        return D.Categorical(logits=policy / self.temperature)


class GaussianSampler(Sampler):
    def __init__(self, act_dim: int):
        super().__init__()
        self.logstd = nn.Parameter(torch.zeros(act_dim, device=DEVICE))

    def get_dist(self, policy: torch.Tensor) -> D.Distribution:
        return D.Normal(policy, self.logstd.exp())


def gen_sampler(sampler_args: dict):
    sampler = eval(f"{sampler_args['type']}Sampler")(**sampler_args["args"])
    return sampler
