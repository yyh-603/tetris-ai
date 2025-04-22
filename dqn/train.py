from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from TetrisWrapper import MyTetrisEnv

import torch
import random
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.manual_seed(114514)
else:
    torch.manual_seed(114514)

random.seed(114514)

env = DummyVecEnv([lambda: Monitor(MyTetrisEnv()) for _ in range(16)])

tmp_path = "./logs"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = DQN("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

try:
    for epoch in tqdm(range(100)):
        model.learn(total_timesteps=100_000, progress_bar=True)
        model.save(f"model/dqn_{epoch}")
except Exception as e:
    print(e)
finally:
    model.save("model_dqn")