from stable_baselines3 import A2C
import gymnasium as gym
import ale_py
from TetrisWrapper import MyWrapper
import torch

gym.register_envs(ale_py)

env = gym.make('ALE/Tetris-v5', max_episode_steps=-1)
env = MyWrapper(env)

policy = torch.load("bc_policy", weights_only=False)
policy = policy['state_dict']

model = A2C("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": [32, 32]})
model.policy.load_state_dict(policy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
done = False
env = gym.make('ALE/Tetris-v5', render_mode = "human", max_episode_steps=-1)
env = MyWrapper(env)
state, info = env.reset()
while True:
    action = model.predict(state)[0]
    state, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        break
    env.render()