import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
from typing import Iterable

DEVICE = "cuda"