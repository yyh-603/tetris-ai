import gymnasium as gym
import ale_py
from EnvProcess import EnvProcess
import numpy as np

CLEAR_LINE_POINT = 1
ALIVE_POINT = 0

TETROMINO_TYPES = [
    [(0, 0), (0, 1), (1, 0), (1, 1)],  # 0: O
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # 1: I1
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # 2: I2
    [(0, 0), (0, 1), (0, 2), (1, 2)],  # 3: J1
    [(0, 1), (1, 1), (2, 0), (2, 1)],  # 4: J2
    [(0, 0), (1, 0), (1, 1), (1, 2)],  # 5: J3
    [(0, 0), (0, 1), (1, 0), (2, 0)],  # 6: J4
    [(0, 0), (0, 1), (0, 2), (1, 0)],  # 7: L1
    [(0, 0), (0, 1), (1, 1), (2, 1)],  # 8: L2
    [(0, 2), (1, 0), (1, 1), (1, 2)],  # 9: L3
    [(0, 0), (1, 0), (2, 0), (2, 1)],  # 10: L4
    [(0, 1), (0, 2), (1, 0), (1, 1)],  # 11: S1
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # 12: S2
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # 13: Z1
    [(0, 1), (1, 0), (1, 1), (2, 0)],  # 14: Z2
    [(0, 0), (0, 1), (0, 2), (1, 1)],  # 15: T1
    [(0, 1), (1, 0), (1, 1), (2, 1)],  # 16: T2
    [(0, 1), (1, 0), (1, 1), (1, 2)],  # 17: T3
    [(0, 0), (1, 0), (1, 1), (2, 0)],  # 18: T4
]

HEIGHT = 22
WIDTH = 10

NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
DOWN = 4


class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(64)  # 4 * 2 * 8
        self.observation_shape = (HEIGHT * WIDTH, )
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.uint8)
        self._current_state = None
    
    def _state_transform(self, state):
        return np.array(state).flatten().astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = EnvProcess.getStateFromImage(obs)
        self._current_state = state
        return self._state_transform(state), info
    
    
    def step(self, action):
        rotate = action // 16
        dir = action % 16 // 8
        move = action % 8
        current_state = self._current_state
        current_state_count = sum(sum(row) for row in current_state)
        clear_reward = 0

        state, terminated, truncated, info = current_state, False, False, dict()
        round_end = False
        
        for i in range(rotate):
            if terminated or round_end:
                break
            obs, reward, terminated, truncated, info = self.env.step(FIRE)
            state = EnvProcess.getStateFromImage(obs)
            state_sum = sum(sum(row) for row in state)
            current_state_count = state_sum
            clear_reward += reward * CLEAR_LINE_POINT

        for i in range(move):
            if terminated or round_end:
                break
            obs, reward, terminated, truncated, info = self.env.step(RIGHT if dir == 1 else LEFT)
            state = EnvProcess.getStateFromImage(obs)
            state_sum = sum(sum(row) for row in state)
            current_state_count = state_sum
            clear_reward += reward * CLEAR_LINE_POINT


        while not terminated and not round_end:
            obs, reward, terminated, truncated, info = self.env.step(DOWN)
            state = EnvProcess.getStateFromImage(obs)
            state_sum = sum(sum(row) for row in state)
            round_end = state_sum != current_state_count
            current_state_count = state_sum
            clear_reward += reward * CLEAR_LINE_POINT


        return self._state_transform(state), clear_reward + ALIVE_POINT, terminated, truncated, info


if __name__ == "__main__":
    # check TETROMINO_TYPES
    for i in range(19):
        print(f"====== Type {i} ======")
        for x in range(0, 4):
            for y in range(0, 4):
                if (x, y) in TETROMINO_TYPES[i]:
                    print(1, end=' ')
                else:
                    print(0, end=' ')
            print()
        
        if sorted(TETROMINO_TYPES[i], key=lambda x: (x[0], x[1])) != TETROMINO_TYPES[i]:
            print("Error: TETROMINO_TYPES is not sorted")


class MyTetrisEnv(gym.Env):
    def __init__(self, render_mode = None):
        super().__init__()
        if render_mode is not None:
            self.env = gym.make("ALE/Tetris-v5", render_mode = render_mode)
        else:
            self.env = gym.make("ALE/Tetris-v5")
        self.env = MyWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_shape = self.env.observation_shape
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def close(self):
        self.env.close()


if __name__ == "__main__":
    gym.register_envs(ale_py)
    from stable_baselines3.common.env_checker import check_env
    env = MyTetrisEnv()
    check_env(env, warn=True)