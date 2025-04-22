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
ROTATE_TYPES = [
    0,              # O
    2, 1,           # I
    4, 5, 6, 3,     # J
    8, 9, 10, 7,    # L
    12, 11,         # S
    14, 13,         # Z
    16, 17, 18, 15  # T
]
ROTATE_POSITION_OFFSET = [
    (0, 0),                             # O
    (-1, 1), (1, -1),                   # I
    (-1, 1), (0, -1), (0, 1), (1, -1),  # J
    (-1, 0), (0, 2), (0, -1), (1, -1),  # L
    (0, 0), (0, 0),                     # S
    (0, 2), (0, -2),                    # Z
    (-1, 1), (0, 0), (0, 0), (1, -1)    # T
]

TETROMINO_SPAWN_AERA = [(x, y) for x in range(1, 5) for y in range(3, 7)]

TETROMINO_SPAWN_POSITION = []

HEIGHT = 22
WIDTH = 10

NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
DOWN = 4

ACTION_SPACE_SIZE = 64

INF = 1e9
EPS = 1e-6

LANDING_PENALTY = -4.500158825082766
ROWS_ELIMINATED_PENALTY = 3.4181268101392694
ROWS_TRANSITION_PENALTY = -3.2178882868487753
COLS_TRANSITION_PENALTY = -9.348695305445199
HOLE_PENALTY = -7.899265427351652
WELL_PENALTY = -3.3855972247263626
SEARCH_DEPTH = 0


class Tetromino:
    def __init__(self, shape: list[tuple[int, int]]):
        """
        self.id: tetromino type
        self.pos: position of the tetromino # first block in TETROMINO_TYPES
        """
        assert len(shape) == 4
        shape = sorted(shape, key=lambda x: (x[0], x[1]))
        self.pos = shape[0]
        
        min_x, min_y = HEIGHT, WIDTH
        for x, y in shape:
            min_x = min(min_x, x)
            min_y = min(min_y, y)

        shape = [(x - min_x, y - min_y) for x, y in shape]
        self.id = 0
        for id, tetromino in enumerate(TETROMINO_TYPES):
            if shape == tetromino:
                self.id = id
                break
    
    def __str__(self):
        ret = ''
        for x in range(4):
            for y in range(4):
                ret += '1 ' if (x, y) in TETROMINO_TYPES[self.id] else '0 '
            ret += '\n'
        return ret

    def expand(self) -> list[tuple[int, int]]:
        dx, dy = self.pos[0] - TETROMINO_TYPES[self.id][0][0], self.pos[1] - TETROMINO_TYPES[self.id][0][1]
        ret = [(x + dx, y + dy) for x, y in TETROMINO_TYPES[self.id]]
        return ret
    
    def check_collision(self, board):
        occupy = self.expand()
        
        # check if the tetromino is out of bounds
        for x, y in occupy:
            if x < 0 or x >= HEIGHT or y < 0 or y >= WIDTH:
                return True
        
        # check if the tetromino is colliding with the board
        for x, y in occupy:
            if board[x][y] != 0:
                return True
        
        return False

    def rotate(self, board):
        prev_pos, prev_id = self.pos, self.id
        self.pos = (self.pos[0] + ROTATE_POSITION_OFFSET[self.id][0], self.pos[1] + ROTATE_POSITION_OFFSET[self.id][1])
        self.id = ROTATE_TYPES[self.id]
        if self.check_collision(board):
            self.pos = prev_pos
            self.id = prev_id
            return False
        return True
    
    def move_left(self, board):
        prev_pos = self.pos
        self.pos = (self.pos[0], self.pos[1] - 1)
        if self.check_collision(board):
            self.pos = prev_pos
            return False
        return True

    def move_right(self, board):
        prev_pos = self.pos
        self.pos = (self.pos[0], self.pos[1] + 1)
        if self.check_collision(board):
            self.pos = prev_pos
            return False
        return True
    
    def move_down(self, board):
        prev_pos = self.pos
        self.pos = (self.pos[0] + 1, self.pos[1])
        if self.check_collision(board):
            self.pos = prev_pos
            return False
        return True

    def drop(self, board):
        while self.move_down(board):
            pass
        return self.pos

class PDAlgorithm:
    def __init__(self):
        pass

    def _extract_next_tetro(self, board):
        
        tetro = []
        for x, y in TETROMINO_SPAWN_AERA:
            if board[x][y] == 1:
                tetro.append((x, y))
        if len(tetro) != 4:
            return None
        
        return Tetromino(tetro)

    def _get_next_states(self, board, tetro = None): 
        if tetro is None:
            tetro = self._extract_next_tetro(board)
        if tetro is None:
            return []
        # print(f'id: {tetro.id}')
        tetro_pos = tetro.expand()
        for x, y in tetro_pos:
            board[x][y] = 0
        
        states = []
        for action in range(ACTION_SPACE_SIZE):
            rotate = action // 16
            dir = action % 16 // 8
            move = action % 8
            new_tetro = Tetromino(tetro.expand())
            ok = True
            for i in range(rotate):
                ok = ok and new_tetro.rotate(board)
            for i in range(move):
                if dir == 0:
                    ok = ok and new_tetro.move_left(board)
                else:
                    ok = ok and new_tetro.move_right(board)
            if ok:
                new_tetro.drop(board)
                # check row elimination
                tmp_board = [[board[x][y] for y in range(WIDTH)] for x in range(HEIGHT)]
                for x, y in new_tetro.expand():
                    tmp_board[x][y] = 1
                new_board = []
                for row in tmp_board:
                    if not all(row):
                        new_board.append(row)
                
                for x in range(HEIGHT - len(new_board)):
                    new_board.insert(0, [0] * WIDTH)

                states.append((action, new_board, new_tetro))
        return states
    
    def _get_landing_height(self, tetro):
        tetro_pos = tetro.expand()
        mean_height = 0
        for x, y in tetro_pos:
            mean_height += HEIGHT - x
        mean_height /= 4
        return mean_height

    def _get_rows_eliminated(self, board, tetro):
        tetro_pos = tetro.expand()
        eliminated_rows = 0
        tetro_point = 0
        for x in range(HEIGHT):
            if all(board[x][y] == 1 for y in range(WIDTH)):
                eliminated_rows += 1
                for y in range(WIDTH):
                    tetro_point += 1 if (x, y) in tetro_pos else 0
        return eliminated_rows * tetro_point

    def _get_rows_transition(self, board):
        ret = 0
        for x in range(HEIGHT):
            cur = 1
            for y in range(WIDTH):
                if board[x][y] != cur:
                    cur = board[x][y]
                    ret += 1
            ret += 1 if cur == 0 else 0
        return ret

    def _get_cols_transition(self, board):
        ret = 0
        for y in range(WIDTH):
            cur = 1
            for x in range(HEIGHT):
                if board[x][y] != cur:
                    cur = board[x][y]
                    ret += 1
            ret += 1 if cur == 0 else 0
        return ret

    def _get_holes(self, board):
        ret = 0
        for x in range(HEIGHT - 1):
            for y in range(WIDTH):
                if board[x][y] == 1 and board[x + 1][y] == 0:
                    ret += 1
        return ret

    def _get_wells(self, board):
        wells = [[0] * WIDTH for _ in range(HEIGHT)]
        for x in range(HEIGHT):
            for y in range(WIDTH):
                if board[x][y] == 1:
                    continue
                if y == 0:
                    if board[x][y + 1] == 1:
                        wells[x][y] = 1                    
                elif y == WIDTH - 1:
                    if board[x][y - 1] == 1:
                        wells[x][y] = 1
                else:
                    if board[x][y - 1] == 1 and board[x][y + 1] == 1:
                        wells[x][y] = 1
        for x in range(1, HEIGHT):
            for y in range(WIDTH):
                wells[x][y] = wells[x - 1][y] + wells[x][y] if wells[x][y] > 0 else 0

        return sum(sum(row) for row in wells)
    
    def get_reward_from_board(self, board):
        tetro = self._extract_next_tetro(board)
        if tetro is None:
            return 0
        tetro_pos = tetro.expand()
        for x, y in tetro_pos:
            board[x][y] = 0
        return self.get_reward(self, board, tetro)

    def get_reward(self, board, tetro, depth = SEARCH_DEPTH):
        if depth == 0:
            return (
                self._get_landing_height(tetro) * LANDING_PENALTY +
                self._get_rows_eliminated(board, tetro) * ROWS_ELIMINATED_PENALTY +
                self._get_rows_transition(board) * ROWS_TRANSITION_PENALTY +
                self._get_cols_transition(board) * COLS_TRANSITION_PENALTY +
                self._get_holes(board) * HOLE_PENALTY +
                self._get_wells(board) * WELL_PENALTY
            )
        states = self._get_next_states(board, tetro)
        
        for action_i, state_i, tetro_i in states:
            reward_i = -INF
            for nxt_tetro in TETROMINO_TYPES:
                nxt_tetro = Tetromino(nxt_tetro)


    def get_action(self, board):
        assert len(board) == HEIGHT and len(board[0]) == WIDTH

        states = self._get_next_states(board)
        if len(states) == 0:
            return -1
        action, state, tetro = states[0]
        reward = self.get_reward(state, tetro)
        
        for i in range(1, len(states)):
            action_i, new_board_i, tetro_i = states[i]
            reward_i = self.get_reward(new_board_i, tetro_i)
            if reward_i > reward:
                action, reward = action_i, reward_i
            elif abs(reward_i - reward) < EPS:
                pro_i = (action_i % 8) * 100 + action_i // 16
                pro = (action % 8) * 100 + action // 16
                if pro_i < pro:
                    action, reward = action_i, reward_i
        # print(f"Action: {action}, Reward: {reward}")
        return action


if __name__ == '__main__':
    tetro = [Tetromino(tetro) for tetro in TETROMINO_TYPES]
    for t in tetro:
        print(f'====== {t.id} ======')
        print(t.pos)
        print(t)
        t.rotate()
        print(t.pos)
        print(t)