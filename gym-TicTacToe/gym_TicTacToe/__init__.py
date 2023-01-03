from gym.envs.registration import register

register(
    id='gym_TicTacToe/TicTacToe-v0',
    entry_point='gym_TicTacToe.envs:TicTacToeEnv',
    max_episode_steps=9,
)