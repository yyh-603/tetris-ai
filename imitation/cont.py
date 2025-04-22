from stable_baselines3 import A2C
import gymnasium as gym
import ale_py
from TetrisWrapper import MyWrapper
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

gym.register_envs(ale_py)

def make_env():
    env = gym.make('ALE/Tetris-v5')
    env = MyWrapper(env)
    obs, info = env.reset()
    return env

env = gym.make('ALE/Tetris-v5', max_episode_steps=-1)
env = MyWrapper(env)
env = DummyVecEnv([make_env] * 8)

policy = torch.load("bc_policy", weights_only=False)
policy = policy['state_dict']

logger = configure("logs", ["stdout", "csv", "tensorboard"])
model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": [32, 32]})
model.policy.load_state_dict(policy)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
done = False

for id in tqdm(range(20)):
    model.learn(total_timesteps=25000, progress_bar=True, tb_log_name="cont")
    model.save(f"model_{id}")