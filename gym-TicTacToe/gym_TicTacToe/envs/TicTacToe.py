import gym
from gym import spaces
import pygame
import numpy as np
from numpy.random import default_rng


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=3):
        #assert size == 3, "Only supporting a size of 3 currently."
        self.size = size  # The size of the square grid. 
        self.window_size = 512  # The size of the PyGame window
        self.rng = default_rng()

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.size * self.size,), dtype=np.int8
        )

        # When self.size = 3, we have 9 actions, corresponding to each space on the 3x3 board.
        self.action_space = spaces.Discrete(self.size*self.size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        return self.board #{"players_turn": self.players_turn, "board": self.board}

    def _get_info(self):
        return {"turn_number": self.turn_count, "players_turn": self.players_turn}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.board = np.zeros(self.size*self.size, np.int8) # Init board as 3x3 grid of zeros
        self.players_turn = [1, -1][self.rng.integers(2)]  # Randomly pick first player
        self.player_dict = {1: "X", -1: "O"}
        self.turn_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _check_board(self):
        grid_board = self.board.reshape(3,3)
        rows, cols = grid_board.shape
        assert rows == cols, "check_board() assumes a square board."
        for row in range(rows):
            if abs(sum(grid_board[row])) == rows:
                return 1
            if abs(sum(grid_board[:, row])) == rows:
                return 1
        if abs(sum(grid_board.diagonal())) == rows:
            return 1
        if abs(sum(np.flip(grid_board, 0).diagonal())) == rows:
            return 1
        return 0

    def get_available_moves(self):
        return np.where(self.board == 0)[0]

    def get_random_move(self):
        num_moves = len(self.get_available_moves())
        if num_moves == 0:
            return 0
        return self.get_available_moves()[self.rng.integers(num_moves)]
    
    def step(self, action):

        available_moves = self.get_available_moves()
        num_moves = len(available_moves)

        if action in available_moves:
            self.board[action] = self.players_turn
        else:
            print("Invalid move:", action)

        # if num_moves != 0:
        #     random_choice = self.rng.integers(num_moves)
        #     self.board[available_moves[0][random_choice], available_moves[1][random_choice]] = player

        # Map the action (0-8) to the coesponding location coordinates (e.g. 0 -> [0,0]) 
        #location = self._action_to_location[action]

        
        # Player wins 
        if self._check_board() == 1: 
            terminated = True
            reward = 2
        # Draw
        elif num_moves == 1: 
            terminated = True
            reward = 1
        else:
            terminated = False
            reward = 0

        # Alternate player turn
        self.players_turn = self.players_turn * -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # First we draw the board guidelines
        for x in range(1, self.size + 1):
            pygame.draw.line(canvas, 0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(canvas, 0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Convert vector version of the board to 2D
        grid_board = self.board.reshape(self.size, self.size)
        
        # Next, draw markers ('X', 'O') if present 
        taken_spaces = np.where(grid_board != 0)
        for i in range(len(taken_spaces[0])):
            location = np.array([taken_spaces[0][i], taken_spaces[1][i]])

            # "X"
            if grid_board[location[0], location[1]] == 1:
                box_min = (location + 0.15) * pix_square_size
                box_max = (location + 0.85) * pix_square_size
                pygame.draw.line(canvas, (255, 0, 0),
                    box_min, box_max,
                    width=10,
                )
                pygame.draw.line(canvas, (255, 0, 0),
                    [box_min[0], box_max[1]], [box_max[0], box_min[1]],
                    width=10,
                )
            # "O"
            else:
                pygame.draw.circle(canvas, (0, 0, 255),
                    (location + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()