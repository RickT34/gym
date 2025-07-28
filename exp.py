from env import *
from agent import *
from utils import *
from tqdm import tqdm
from logger import Logger

ENVARGS = {
    "id": "LunarLander-v3",
    "continuous": False,
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 15.0,
}
BATCH_SIZE = 64
EPOCHS = 200

env = gym.make(**ENVARGS)

logger = Logger(ENVARGS["id"])

agent = PolicyGradientAgent()

def human_eval():
    env2 = gym.make(render_mode="rgb_array", **ENVARGS)
    env2 = gym.wrappers.RenderCollection(env2)
    Trajectory.sample_from_agent(agent, env2)
    flames = env2.render()
    logger.log_video("video", flames)

def train_loop():
    for epoch in tqdm(range(EPOCHS)):
        trajectories = [Trajectory.sample_from_agent(agent, env) for _ in range(BATCH_SIZE)]
        loss = agent.update(trajectories)
        logger.log_scalar("loss", loss)
        if epoch % 10 == 0:
            human_eval()
        logger.commit()

if __name__ == "__main__":
    train_loop()
    human_eval()
    logger.commit()
    