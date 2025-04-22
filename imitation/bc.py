import gymnasium as gym
import ale_py
import numpy as np
from imitation.data.types import Transitions
from imitation.algorithms import bc
from imitation.util.logger import configure
from pdalgo import PDAlgorithm
from TetrisWrapper import MyWrapper
from tqdm import tqdm
import torch



if __name__ == '__main__':
    gym.register_envs(ale_py)
    env = gym.make('ALE/Tetris-v5', max_episode_steps=-1)
    env = MyWrapper(env)
    agent = PDAlgorithm()


    state, info = env.reset()

    DATA_COUNT = 100000

    states = []
    actions = []
    infos = []
    nxt_states = []
    dones = []

    for step in tqdm(range(DATA_COUNT)):
        action = agent.get_action(state)
        record = True
        if action == -1:
            record = False
            action = 0
        nxt_state, rewards, terminated, truncated, info = env.step(action)
        if record:
            states.append(state)
            actions.append(action)
            nxt_states.append(nxt_state)
            infos.append(info)
            dones.append(terminated)

        if step % 1000 == 0:
            state, info = env.reset()

        state = nxt_state
    print(len(states))

    trans = Transitions(
        obs=np.array(states),
        acts=np.array(actions),
        infos=np.array(infos),
        next_obs=np.array(nxt_states),
        dones=np.array(dones),
    )
    log_dir = "logs"
    custom_logger = configure(log_dir, ["tensorboard", "stdout"])
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,  
        demonstrations=trans,
        rng=np.random.default_rng(),
        custom_logger=custom_logger,
    )
    epoch = 0
    try:
        def on_epoch_end():
            global epoch
            epoch += 1
            print(f"Epoch {epoch} finished")
            bc_trainer.policy.save(f"model4/bc_policy_{epoch}.pt")
        bc_trainer.train(n_epochs=300, on_epoch_end=on_epoch_end)

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        bc_trainer.policy.save('bc_policy.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = False
    env = gym.make('ALE/Tetris-v5', render_mode = "human", max_episode_steps=-1)
    env = MyWrapper(env)
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)
    state = state.flatten()
    state = state.unsqueeze(0)
    while True:
        action = bc_trainer.policy(state)
        state, rewards, terminated, truncated, info = env.step(action)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.flatten()
        state = state.unsqueeze(0)
        if terminated:
            break
        env.render()