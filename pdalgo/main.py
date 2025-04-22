import gymnasium as gym
import ale_py
import torch
from TetrisWrapper import MyWrapper
import random
from algo import PDAlgorithm

TRAINING = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(114514)
else:
    torch.manual_seed(114514)
random.seed(114514)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gym.register_envs(ale_py)
    env = gym.make('ALE/Tetris-v5', render_mode = "human", max_episode_steps=-1)
    env = MyWrapper(env)

    agent = PDAlgorithm()
    
    state, info = env.reset()
    
    total_reward = 0
    total_step = 0
    frame_id = 0
    while True:
        action = agent.get_action(state)

        # Human input
        # action = list(map(int, input().split()))
        # action = action[0] * 16 + action[1] * 8 + action[2]

        state, rewards, terminated, truncated, info = env.step(action)
        total_reward += rewards
        total_step += 1
        frame_id += 1
        env.render()
        print(f"Frame: {frame_id}, Reward: {rewards}, Total reward: {total_reward}, Total step: {total_step}")