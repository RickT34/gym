from env import DEVICE
from utils import *
import model
import gymnasium as gym
from typing import Iterable, Callable
import torch
from logger import Logger
from tqdm import tqdm
import itertools

def run_env(env:gym.Env, agent:"Agent", agent_args:dict, step_trigger:Callable|None=None, batch_size:int=1):
    for batch in range(batch_size):
        obs, info = env.reset()
        step = 0
        while True:
            obs = to_tensor(obs)
            act = agent.get_action(obs, **agent_args)
            r = env.step(to_numpy(act))
            if step_trigger is not None:
                step_trigger(step, obs, act, r)
            next_obs, rew, done, fail, info = r
            if done or fail:
                break
            obs = next_obs
            step += 1

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
        agent: "Agent", env: gym.Env, sample: bool = True, batch_size: int = 1
    ) -> list["Trajectory"]:
        re = []
        traj = None
        def step_trigger(step, obs, act, r):
            next_obs, rew, done, fail, info = r
            nonlocal traj
            if step == 0:
                traj = Trajectory()
            assert traj is not None
            traj.append(obs, act, rew)
            if done or fail:
                traj.done = done
                re.append(traj)
        run_env(env, agent, {"sample": sample}, step_trigger=step_trigger, batch_size=batch_size)
        return re

    def __iter__(self):
        return zip(self.observatons, self.actions, self.rewards)

def generate_flames(envargs:dict, agent:"Agent", sample:bool=False):
    env = gym.make(render_mode="rgb_array", **envargs)
    env = gym.wrappers.RenderCollection(env)
    def step_trigger(step, obs, act, r):
        next_obs, rew, done, fail, info = r
        print(f"Step {step}:")
        print(f"  Observation: {obs}")
        print(f"  Action: {act}")
        print(f"  Reward: {rew}")
    run_env(env, agent, {"sample": sample})
    flames = env.render()
    return flames

class Agent:
    def __init__(self, policy, discrete:bool):
        self.policy = policy
        self.discrete = discrete
        self.policy_logstd = 0.0

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        output = self.policy(obs)
        if sample:
            if self.discrete:
                dist = torch.distributions.Categorical(logits=output)
            else:
                dist = torch.distributions.Normal(output,torch.exp(self.policy_logstd)) # type: ignore
            action = dist.sample()
        else:
            if self.discrete:
                action = output.argmax(dim=-1)
            else:
                action = output
        return action

    def update(self, trajectories: list[Trajectory]):
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
        discrete: bool,
        model_path: str | None = None,
        **kwargs
    ):
        policy = model.build_mlp(ob_dim, layers)
        if model_path is not None:
            policy.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=DEVICE)
            )
        super().__init__(policy, discrete)
        if not discrete:
            self.policy_logstd = torch.nn.Parameter(torch.zeros((layers[-1],), device=DEVICE), requires_grad=True)
            self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), [self.policy_logstd]), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def update(self, trajectories: list[Trajectory]):
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
            output = self.policy(obs)
            if self.discrete:
                log_prob = torch.distributions.Categorical(logits=output).log_prob(act)
            else:
                log_prob = torch.distributions.Normal(output, torch.exp(self.policy_logstd)).log_prob(act).sum(dim=-1)
            loss = -log_prob * rw
            losses.append(loss.sum())
        losses = torch.stack(losses)
        loss = losses.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return losses.detach()

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
            losses = self.update(Trajectory.sample_from_agent(self, env, batch_size=batch_size))
            logger.log_scalar("loss_mean", losses.mean().item())
            logger.log_scalar("loss_std", losses.std().item())
            epo_trigger(epoch)
            logger.commit()
