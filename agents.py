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
            obs_t = torch.tensor(obs, device=DEVICE)
            act = agent.get_action(obs_t, **agent_args)
            if len(act.shape) == 0:
                act_e = act.item()
            else:
                act_e = to_numpy(act)
            r = env.step(act_e)
            if step_trigger is not None:
                step_trigger(step, obs_t, act, r)
            next_obs, rew, done, fail, info = r
            if done or fail:
                break
            obs = next_obs
            step += 1

def run_env_vec(env:gym.vector.VectorEnv, agent:"Agent", agent_args:dict, max_steps:int, step_trigger:Callable|None=None):
    obs, info = env.reset()
    for step in range(max_steps):
        obs = to_tensor(obs)
        act = agent.get_action(obs, **agent_args)
        r = env.step(to_numpy(act))
        if step_trigger is not None:
            step_trigger(step, obs, act, r)
        obs = r[0]

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

def debug_step_trigger(step, obs, act, r):
    next_obs, rew, done, fail, info = r
    if (fail | done).any():
        print(f"Step {step}:")
        print(f"  Observation: {obs}")
        print(f"  Action: {act}")
        print(f"  Return: {r}")

def generate_flames(envargs:dict, agent:"Agent", sample:bool=False):
    env = gym.make(render_mode="rgb_array", **envargs)
    env = gym.wrappers.RenderCollection(env)
    run_env(env, agent, {"sample": sample})
    flames = env.render()
    return flames

def trajectories_logging(trajectories: list[Trajectory], logger: Logger):
    steps_tot = 0
    reward_tot = 0.0
    reward_max = -float("inf")
    reward_min = float("inf")
    for traj in trajectories:
        steps_tot += len(traj)
        rews = np.stack(traj.rewards)
        reward_tot += rews.sum()
        reward_max = max(reward_max, rews.max())
        reward_min = min(reward_min, rews.min())
    logger.log_scalar("reward_mean", reward_tot / len(trajectories))
    logger.log_scalar("reward_max", reward_max)
    logger.log_scalar("reward_min", reward_min)
    logger.log_scalar("eps_len", steps_tot/len(trajectories))

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

    def save(self, path: str):
        raise NotImplementedError()

    def training_loop(
        self, env: gym.vector.VectorEnv, logger: Logger, epo_trigger: Callable, **kwargs
    ):
        raise NotImplementedError()


class PolicyGradientAgent(Agent):
    def __init__(
        self,
        act_dim: int,
        lr: float,
        gamma: float,
        discrete: bool,
        policy: list[dict],
        model_path: str | None = None,
        **kwargs
    ):
        net = model.gen_policy(policy)
        if model_path is not None:
            net.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=DEVICE)
            )
        super().__init__(net, discrete)
        if not discrete:
            self.policy_logstd = torch.nn.Parameter(torch.zeros((act_dim,), device=DEVICE), requires_grad=True)
            self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), [self.policy_logstd]), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = torch.tensor(gamma, device=DEVICE)

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

    def update_vec(self, env:gym.vector.VectorEnv, logger:Logger, max_steps:int):
        n = env.num_envs
        losses = torch.zeros(n, device=DEVICE)
        zeros = torch.zeros(n, device=DEVICE)
        probs = torch.zeros(n, device=DEVICE)
        gammas = torch.ones(n, device=DEVICE)
        rew_tot = 0
        rew_max = -float("inf")
        rew_min = float("inf")
        tot = n // 2
        def update_step(step, obs, act, r):
            nonlocal losses, probs, gammas, rew_tot, tot, rew_max, rew_min
            next_obs, rew, done, fail, info = r
            rew_tot += rew.sum()
            rew_max = max(rew_max, rew.max())
            rew_min = min(rew_min, rew.min())
            rew = to_tensor(rew)
            done = torch.from_numpy(done).to(DEVICE)
            fail = torch.from_numpy(fail).to(DEVICE)
            
            rst = done | fail
            output = self.policy(obs)
            if self.discrete:
                log_prob = torch.distributions.Categorical(logits=output).log_prob(act)
            else:
                log_prob = torch.distributions.Normal(output, torch.exp(self.policy_logstd)).log_prob(act).sum(dim=-1)
            probs = log_prob + torch.where(rst, zeros, probs*self.gamma)
            losses -= probs*rew
            tot += rst.sum().item()
            
        run_env_vec(env, self, {"sample": True}, max_steps, step_trigger=update_step)
        loss = losses.sum() / tot
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.log_scalar("score_mean", loss.item())
        logger.log_scalar("eps_len", n*max_steps / tot)
        logger.log_scalar("reward_mean", rew_tot / (n*max_steps))
        logger.log_scalar("reward_max", rew_max)
        logger.log_scalar("reward_min", rew_min)
        

    def training_loop(
        self,
        env: gym.vector.VectorEnv,
        logger: Logger,
        epo_trigger: Callable,
        *,
        epochs: int,
        max_steps: int,
        **kwargs
    ):
        for epoch in tqdm(range(epochs)):
            self.update_vec(env, logger, max_steps)
            epo_trigger(epoch)
            logger.commit()
