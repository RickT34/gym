from env import *

def to_tensor(x:np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().to(DEVICE)

def to_numpy(x:torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()