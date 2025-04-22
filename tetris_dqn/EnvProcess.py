TARGET_POINT_X = [i for i in range(4, 180, 8)]
TARGET_POINT_Y = [i for i in range(3, 42, 4)]
OBS_XSIZE = len(TARGET_POINT_X) # 22
OBS_YSIZE = len(TARGET_POINT_Y) # 10
BACKGROUND_COLOR = (111, 111, 111)

class EnvProcess:

    '''
    Given an Image, return the observation.
    Input image size should be (210, 160, 3)
    '''
    def getStateFromImage(img):
        assert img.shape == (210, 160, 3), "Image size should be (210, 160, 3)"
        
        # Image Preprocessing
        img = img[20:]
        img = img[7:182, 22:63]

        # Create observation map
        obs = [[0 for _ in range(OBS_YSIZE)] for _ in range(OBS_XSIZE)]
        for mp_i, i in enumerate(TARGET_POINT_X):
            for mp_j, j in enumerate(TARGET_POINT_Y):
                if img[i][j][0] == 111 and img[i][j][1] == 111 and img[i][j][2] == 111:
                    obs[mp_i][mp_j] = 0
                else:
                    obs[mp_i][mp_j] = 1
        return obs


if __name__ == "__main__":
    # Test the EnvProcess class
    import gymnasium as gym
    import ale_py
    
    gym.register_envs(ale_py)
    env = gym.make('ALE/Tetris-v5', render_mode='human')
    state, info = env.reset()
    state = EnvProcess.getStateFromImage(state)

    TEST_COUNT = 100

    for _ in range(TEST_COUNT):
        action = env.action_space.sample()
        state, rewards, terminated, truncated, info = env.step(action)
        state = EnvProcess.getStateFromImage(state)

        if terminated or truncated:
            state, info = env.reset()
            state = EnvProcess.getStateFromImage(state)
        
        env.render()
        with open("test.txt", "w+") as f:
            for i in range(OBS_XSIZE):
                for j in range(OBS_YSIZE):
                    f.write(f"{state[i][j]} ")
                f.write("\n")
        input("Next State...")
    env.close()
    