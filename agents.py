from env import DEVICE
from utils import *
import model
import gymnasium as gym
from typing import Iterable, Callable
import torch
from logger import Logger
from tqdm import tqdm
import itertools


def run_env(
    env: gym.Env,
    agent: "Agent",
    agent_args: dict = {},
    step_trigger: Callable | None = None,
    batch_size: int = 1,
):
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


def run_env_vec(
    env: gym.vector.VectorEnv,
    agent: "Agent",
    max_steps: int,
    agent_args: dict = {},
    step_trigger: Callable | None = None,
):
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
        agent: "Agent", env: gym.Env, batch_size: int = 1
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

        run_env(env, agent, step_trigger=step_trigger, batch_size=batch_size)
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


def generate_flames(envargs: dict, agent: "Agent"):
    env = gym.make(render_mode="rgb_array", **envargs)
    env = gym.wrappers.RenderCollection(env)
    run_env(env, agent)
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
    logger.log_scalar("eps_len", steps_tot / len(trajectories))


class Agent:
    def __init__(self, policy_net: torch.nn.Module, policy_sampler: model.Sampler):
        self.policy_net = policy_net
        self.policy_sampler = policy_sampler

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        output = self.policy_net(obs)
        dist = self.policy_sampler.get_dist(output)
        action = dist.sample()
        return action

    def save(self, path: str):
        raise NotImplementedError()

    def training_loop(
        self, env: gym.vector.VectorEnv, logger: Logger, epo_trigger: Callable, **kwargs
    ):
        raise NotImplementedError()

    @staticmethod
    def from_config(config: dict) -> "Agent":
        agent = eval(config["type"])(**config["args"])
        assert isinstance(agent, Agent)
        return agent

    def load(self, path: str):
        raise NotImplementedError()


class PolicyGradientAgent(Agent):
    def __init__(
        self, lr: float, gamma: float, epochs: int, max_steps: int, arch: dict
    ):
        policy_net = model.gen_net(arch["policy"])
        policy_sampler = model.gen_sampler(arch["sampler"])
        super().__init__(policy_net, policy_sampler)
        self.optimizer = torch.optim.Adam(
            itertools.chain(policy_net.parameters(), policy_sampler.parameters()), lr=lr
        )
        self.gamma = torch.tensor(gamma, device=DEVICE)
        self.epochs = epochs
        self.max_steps = max_steps

    def save(self, path: str):
        torch.save(
            (self.policy_net.state_dict(), self.policy_sampler.state_dict()), path
        )

    def load(self, path: str):
        net_state_dict, sampler_state_dict = torch.load(
            path, map_location=DEVICE, weights_only=True
        )
        self.policy_net.load_state_dict(net_state_dict)
        self.policy_sampler.load_state_dict(sampler_state_dict)

    def update(self, env: gym.vector.VectorEnv, logger: Logger, max_steps: int):
        n = env.num_envs
        losses = torch.zeros(n, device=DEVICE)
        zeros = torch.zeros(n, device=DEVICE)
        probs = torch.zeros(n, device=DEVICE)
        gammas = torch.ones(n, device=DEVICE)
        rew_tot = 0
        rew_max = -float("inf")
        rew_min = float("inf")
        tot = n // 2
        rst = torch.zeros(n, device=DEVICE, dtype=torch.bool)

        def update_step(step, obs, act, r):
            nonlocal losses, probs, gammas, rew_tot, tot, rew_max, rew_min, rst
            next_obs, rew, done, fail, info = r
            rew_tot += rew.sum()
            rew_max = max(rew_max, rew.max())
            rew_min = min(rew_min, rew.min())
            rew = to_tensor(rew)

            output = self.policy_net(obs)
            log_prob = self.policy_sampler.get_dist(output).log_prob(act)
            while len(log_prob.shape) >= 2:
                log_prob = log_prob.sum(-1)
            probs = log_prob + torch.where(rst, zeros, probs * self.gamma)
            losses -= probs * rew

            done = torch.from_numpy(done).to(DEVICE)
            fail = torch.from_numpy(fail).to(DEVICE)

            rst = done | fail
            tot += rst.sum().item()

        run_env_vec(env, self, max_steps=max_steps, step_trigger=update_step)
        loss = losses.sum() / tot
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.log_scalar("score_mean", loss.item())
        logger.log_scalar("eps_len", n * max_steps / tot)
        logger.log_scalar("reward_mean", rew_tot / (n * max_steps))
        logger.log_scalar("reward_max", rew_max)
        logger.log_scalar("reward_min", rew_min)

    def training_loop(
        self,
        env: gym.vector.VectorEnv,
        logger: Logger,
        epo_trigger: Callable,
        **kwargs,
    ):
        self.policy_net.train()
        self.policy_sampler.train()
        for epoch in tqdm(range(self.epochs)):
            self.update(env, logger, self.max_steps)
            epo_trigger(epoch)
            logger.commit()


class ActorCriticAgent(Agent):
    def __init__(
        self, lr: float, gamma: float, epochs: int, max_steps: int, arch: dict
    ):
        policy_net = model.gen_net(arch["policy"])
        policy_sampler = model.gen_sampler(arch["sampler"])
        super().__init__(policy_net, policy_sampler)
        self.optimizer_policy = torch.optim.Adam(
            itertools.chain(policy_net.parameters(), policy_sampler.parameters()), lr=lr
        )

        self.value_net = model.gen_net(arch["value"])
        self.optimizer_value = torch.optim.Adam(
            self.value_net.parameters(), lr=lr
        )
        
        self.gamma = torch.tensor(gamma, device=DEVICE)
        self.epochs = epochs
        self.max_steps = max_steps
        
    def save(self, path: str):
        torch.save(
            (self.policy_net.state_dict(), self.policy_sampler.state_dict(), self.value_net.state_dict()), path
        )

    def load(self, path: str):
        net_state_dict, sampler_state_dict, value_state_dict = torch.load(
            path, map_location=DEVICE, weights_only=True
        )
        self.policy_net.load_state_dict(net_state_dict)
        self.policy_sampler.load_state_dict(sampler_state_dict)
        self.value_net.load_state_dict(value_state_dict)

    def update(self, env: gym.vector.VectorEnv, logger: Logger, max_steps: int):
        n = env.num_envs
        rew_tot = 0
        rew_max = -float("inf")
        rew_min = float("inf")
        tot = n // 2
        keep = torch.ones(n, device=DEVICE, dtype=torch.bool)

        score_tot = 0.0
        value_loss_tot = 0.0

        def update_step(step, obs, act, r):
            nonlocal rew_tot, tot, rew_max, rew_min, keep, score_tot, value_loss_tot
            next_obs, rew, done, fail, info = r
            
            rew = to_tensor(rew)[keep]
            rew_tot += rew.sum().item()
            rew_max = max(rew_max, rew.max().item())
            rew_min = min(rew_min, rew.min().item())
            obs = obs[keep]
            act = act[keep]
            next_obs = to_tensor(next_obs)[keep]

            output = self.policy_net(obs)
            log_prob = self.policy_sampler.get_dist(output).log_prob(act)
            while len(log_prob.shape) >= 2:
                log_prob = log_prob.sum(-1)

            rew.unsqueeze_(1)
            log_prob.unsqueeze_(1)

            #Update Value Net
            
            value_x = self.value_net(obs)
            value_y = rew + self.gamma * self.value_net(next_obs)
            value_loss = torch.nn.functional.mse_loss(value_x, value_y)
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
            value_loss_tot += value_loss.item()

            #Update Policy Net
            advantage = rew + self.gamma * self.value_net(next_obs) - self.value_net(obs)
            score = -log_prob * advantage
            score = score.sum()
            self.optimizer_policy.zero_grad()
            score.backward()
            self.optimizer_policy.step()
            score_tot += score.item()


            done = torch.from_numpy(done).to(DEVICE)
            fail = torch.from_numpy(fail).to(DEVICE)
            keep = ~(done | fail)
                
            tot += (~keep).sum().item()

        run_env_vec(env, self, max_steps=max_steps, step_trigger=update_step)
        logger.log_scalar("eps_len", n * max_steps / tot)
        logger.log_scalar("reward_mean", rew_tot / (n * max_steps))
        logger.log_scalar("reward_max", rew_max)
        logger.log_scalar("reward_min", rew_min)
        logger.log_scalar("score_tot", score_tot)
        logger.log_scalar("value_loss_tot", value_loss_tot)

    def training_loop(
        self,
        env: gym.vector.VectorEnv,
        logger: Logger,
        epo_trigger: Callable,
        **kwargs,
    ):
        self.policy_net.train()
        self.policy_sampler.train()
        self.value_net.train()
        for epoch in tqdm(range(self.epochs)):
            self.update(env, logger, self.max_steps)
            epo_trigger(epoch)
            logger.commit()