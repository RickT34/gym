#!/home/rickt/.conda/envs/rlgym/bin/python
from env import *
import agents
from utils import *
from logger import Logger
import argparse
import gymnasium as gym
import datetime
import os


def do_human_eval(env_config, ag: agents.Agent):
    env_config["render_mode"] = "human"
    en = gym.make(**env_config)
    while True:
        agents.run_env(en, ag)


def do_train(label: str, env_config, ag, agent_config):

    en = gym.make_vec(**env_config, num_envs=CORES, vectorization_mode="async")

    logger = Logger(label)
    logger.log_text("env_config", yaml.dump(env_config))
    logger.log_text("agent_config", yaml.dump(agent_config))
    
    save_path = f"data/{label}"

    def epo_rec(epoch: int):
        if epoch % 10 == 9:
            flames = agents.generate_flames(env_config, ag)
            logger.log_video("video", flames, en.metadata["render_fps"])
        if epoch % 100 == 99:
            os.makedirs(save_path, exist_ok=True)
            ag.save(save_path + f"/{epoch}.pth")

    print("Start training...")
    ag.training_loop(en, logger, epo_rec)
    print("Training finished.")

    ag.save(save_path + f"/final.pth")
    print(f"Model saved to {save_path}")


def main(env: str, agent: str, model_path: None | str = None, e: bool = False):
    env_config = read_env_config(env)
    agent_config = read_agent_config(env, agent)

    if model_path is not None:
        agent_config["model_path"] = model_path

    ag = agents.Agent.from_config(agent_config)

    assert isinstance(ag, agents.Agent)

    if e:
        do_human_eval(env_config, ag)
    else:
        label = f"{agent}_{env}+{datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')}"
        do_train(label, env_config, ag, agent_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, help="Name of the environment", default="LunarLander-v3"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Name of the agent",
        choices=AGENT_CHOICES,
        default="pg",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "-e", action="store_true", help="Evaluate the agent", default=False
    )

    args = parser.parse_args()
    main(**{k: v for k, v in args._get_kwargs() if v is not None})
