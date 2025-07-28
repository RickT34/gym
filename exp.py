#!/home/rickt/.conda/envs/rlgym/bin/python
from env import *
import agent
from utils import *
from logger import Logger
import argparse
import gymnasium as gym
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, required=True, help="Name of the environment",
        choices=ENV_CHOICES
    )
    parser.add_argument("--agent", type=str, required=True, help="Name of the agent", choices=AGENT_CHOICES)
    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    env_config = read_env_config(args.env)
    agent_config = read_agent_config(args.env, args.agent)

    if args.model_path is not None:
        agent_config["model_path"] = args.model_path

    env_config['id'] = args.env
    ag = eval(f"agent.{args.agent}(**agent_config)")

    assert isinstance(ag, agent.Agent)

    label = f"{args.agent}_{args.env}"

    logger = Logger(label)

    env = gym.make(**env_config)

    def epo_rec(eopch: int):
        if eopch % 10 == 9:
            flames = agent.generate_flames(env_config, ag)
            logger.log_video("video", flames)

    print("Start training...")
    ag.training_loop(env, logger, epo_rec, **agent_config)
    print("Training finished.")

    save_path = f'data/{label}+{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.pth'
    ag.save(save_path)
    print(f"Model saved to {save_path}")
