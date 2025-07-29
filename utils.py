import numpy as np
import torch
from env import DEVICE
import yaml

def to_tensor(x:np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(DEVICE)

def to_numpy(x:torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()

def read_env_config(env_name:str) -> dict:
    with open(f"hparams/envs/{env_name}.yaml") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def read_agent_config(env_name:str, agent_name:str) -> dict:
    with open(f"hparams/agents/{agent_name}/{env_name}.yaml") as f:
        return yaml.load(f, Loader=yaml.FullLoader)