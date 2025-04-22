import gymnasium as gym
from EnvProcess import EnvProcess
import torch

DIFF_PENALTY = -0.00003
TOTAL_HEIGHT_PENALTY = -0.000001
MAX_HEIGHT_PENALTY = -0.0001
HOLE_PENALTY = -0.01
ALIVE_POINT = 1
CLEAR_LINE_POINT = 1

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

def get_reward_from_state(state):
    # BFS from buttom
    height, width = HEIGHT, WIDTH
    visited = [[False] * width for _ in range(height)]
    min_height = [height] * width
    one_count = 0
    queue = []
    for i in range(width):
        if state[height-1][i] == 1:
            queue.append((height-1, i))
            visited[height-1][i] = True
    
    while queue:
        cur_x, cur_y = queue.pop(0)
        min_height[cur_y] = min(min_height[cur_y], cur_x)
        one_count += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = cur_x + dx, cur_y + dy
            if 0 <= new_x < height and 0 <= new_y < width and not visited[new_x][new_y] and state[new_x][new_y] == 1:
                queue.append((new_x, new_y))
                visited[new_x][new_y] = True
    
    min_height = [height - i for i in min_height]
    sum_height = sum(min_height)
    diff = sum(abs(min_height[i - 1] - min_height[i]) for i in range(1, len(min_height)))
    hole_count = height * width - sum(min_height) - one_count

    return sum_height * TOTAL_HEIGHT_PENALTY + max(min_height) * MAX_HEIGHT_PENALTY + diff * DIFF_PENALTY + hole_count * HOLE_PENALTY

def state_transform(state):
    # BFS from buttom
    height, width = state.shape
    visited = [[False] * width for _ in range(height)]
    min_height = [height] * width
    one_count = 0
    queue = []
    for i in range(width):
        if state[height-1][i] == 1:
            queue.append((height-1, i))
            visited[height-1][i] = True
    
    while queue:
        cur_x, cur_y = queue.pop(0)
        min_height[cur_y] = min(min_height[cur_y], cur_x)
        one_count += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = cur_x + dx, cur_y + dy
            if 0 <= new_x < height and 0 <= new_y < width and not visited[new_x][new_y] and state[new_x][new_y] == 1:
                queue.append((new_x, new_y))
                visited[new_x][new_y] = True
    
    new_state = [0] * len(TETROMINO_TYPES) + [(height - i) / height for i in min_height]
    nxt_tetromino = []
    for x in range(height):
        for y in range(width):
            if state[x][y] == 1 and not visited[x][y]:
                nxt_tetromino.append((x, y))
                
    min_x, min_y = height, width
    for x, y in nxt_tetromino:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
    nxt_tetromino = [(x - min_x, y - min_y) for x, y in nxt_tetromino]
    nxt_tetromino = sorted(nxt_tetromino, key=lambda x: (x[0], x[1]))
    for id, tetromino in enumerate(TETROMINO_TYPES):
        if tetromino == nxt_tetromino:
            # print(f"Found tetromino: {id}")
            new_state[id] = 1
            break
    new_state = torch.tensor(new_state)
    return new_state, id


class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(64)  # 4 * 2 * 8
        self._current_state = None
        self._current_reward = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = EnvProcess.getStateFromImage(obs)
        self._current_state = state
        self._current_reward = 0
        return state, info
    
    
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
            current_state_count = sum(sum(row) for row in state)
            clear_reward += reward * CLEAR_LINE_POINT

        for i in range(move):
            if terminated or round_end:
                break
            obs, reward, terminated, truncated, info = self.env.step(RIGHT if dir == 1 else LEFT)
            state = EnvProcess.getStateFromImage(obs)
            current_state_count = sum(sum(row) for row in state)
            clear_reward += reward * CLEAR_LINE_POINT


        while not terminated and not round_end:
            obs, reward, terminated, truncated, info = self.env.step(DOWN)
            state = EnvProcess.getStateFromImage(obs)
            round_end = (sum(sum(row) for row in state) != current_state_count)
            current_state_count = sum(sum(row) for row in state)
            clear_reward += reward * CLEAR_LINE_POINT

        # total_reward += get_reward_from_state(state)
        # return_reward = state_action_reward - self._current_reward + clear_reward
        self._current_state = state
        # self._current_reward = state_action_reward

        # if not terminated:
        #     total_reward += ALIVE_POINT

        return state, clear_reward, terminated, truncated, info

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