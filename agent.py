from env import DEVICE
from utils import *
import model
import gymnasium as gym
from typing import Iterable, Callable
import torch
from logger import Logger
from tqdm import tqdm


class Trajectory:
    def __init__(self):
        self.observatons: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[np.float64] = []
        self.done = False

    def __len__(self):
        return len(self.observatons)

    def __getitem__(self, idx):
        return self.observatons[idx], self.actions[idx], self.rewards[idx]

    def append(self, obs, act, rew):
        self.observatons.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)

    @staticmethod
    def sample_from_agent(
        agent: "Agent", env: gym.Env, max_steps: int = 1000, sample: bool = True
    ) -> "Trajectory":
        traj = Trajectory()
        obs, info = env.reset()
        for step in range(max_steps):
            obs = to_tensor(obs)
            act = agent.get_action(obs, sample=sample)
            next_obs, rew, done, fail, info = env.step(act.item())
            traj.append(obs, act, rew)
            if done or fail:
                traj.done = done
                break
            obs = next_obs
        return traj

    def __iter__(self):
        return zip(self.observatons, self.actions, self.rewards)

def generate_flames(envargs:dict, agent:"Agent", max_steps:int=1000, sample:bool=False):
    env = gym.make(render_mode="rgb_array", **envargs)
    env = gym.wrappers.RenderCollection(env)
    obs, info = env.reset()
    for step in range(max_steps):
        obs = to_tensor(obs)
        act = agent.get_action(obs, sample=sample)
        next_obs, rew, done, fail, info = env.step(act.item())
        if done or fail:
            break
        obs = next_obs
    flames = env.render()
    return flames

class Agent:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        logits = self.policy(obs)
        if sample:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)
        return action

    def update(self, trajectories: Iterable[Trajectory]):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def training_loop(
        self, env: gym.Env, logger: Logger, epo_trigger: Callable, **kwargs
    ):
        raise NotImplementedError()


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        ob_dim: int,
        layers: list[int],
        lr: float,
        gamma: float,
        model_path: str | None = None,
        **kwargs
    ):
        policy = model.build_mlp(ob_dim, layers)
        if model_path is not None:
            policy.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=DEVICE)
            )
        super().__init__(policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def update(self, trajectories: Iterable[Trajectory]):
        losses = []
        self.policy.train()
        for trajectory in trajectories:
            rtg_sum = 0
            loss = 0
            rtgs = []
            for rw in trajectory.rewards[::-1]:
                rtg_sum = rw + self.gamma * rtg_sum
                rtgs.append(rtg_sum)
            rw = torch.tensor(rtgs[::-1], dtype=torch.float32, device=DEVICE)
            rw -= rw.mean()
            obs = torch.stack(trajectory.observatons)
            act = torch.stack(trajectory.actions)
            logits = self.policy(obs)
            log_prob = torch.distributions.Categorical(logits=logits).log_prob(act)
            loss = -log_prob * rw
            losses.append(loss.sum())
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def training_loop(
        self,
        env: gym.Env,
        logger: Logger,
        epo_trigger: Callable,
        *,
        epochs: int,
        batch_size: int,
        **kwargs
    ):
        for epoch in tqdm(range(epochs)):
            trajectories = [
                Trajectory.sample_from_agent(self, env) for _ in range(batch_size)
            ]
            loss = self.update(trajectories)
            logger.log_scalar("loss", loss)
            epo_trigger(epoch)
            logger.commit()
