from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from TetrisWrapper import MyTetrisEnv

import gymnasium as gym
import ale_py

model = DQN.load("model_dqn")
env = MyTetrisEnv()

total_reward = 0
max_reward = 0

total_step = 0
max_step = 0

for i in range(10):
    obs, _ = env.reset()
    epi_reward = 0
    step = 0
    for frame_id in range(2000):
        action, _ = model.predict(obs)
        obs, rewards, terminated, _, info = env.step(action)
        epi_reward += rewards
        step += 1
        if terminated:
            break
    total_reward += epi_reward
    total_step += step
    max_reward = max(max_reward, epi_reward)
    max_step = max(max_step, step)

    print(epi_reward, step)

print("Max step:", max_step)
print("Mean step:", total_step / 10)
print("Max reward:", max_reward)
print("Mean reward:", total_reward / 10)