from stable_baselines3.common.vec_env import VecVideoRecorder
from TetrisWrapper import MyTetrisEnv

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from stable_baselines3 import DQN

env = MyTetrisEnv(render_mode = "human")
env = VecVideoRecorder(env, "./videos/",
                       record_video_trigger=lambda step: True,
                       video_length=1000,
                       name_prefix="dqn")

model = DQN.load("model_dqn", env=env, custom_objects={'buffer_size': 10000})

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()

env.close()
