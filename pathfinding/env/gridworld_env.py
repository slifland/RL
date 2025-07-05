import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = grid_size * grid_size
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(grid_size, grid_size), dtype=np.uint8
        )
        self.visited = {}
        self.reset(True)
        
        plt.ion()  # turn on interactive mode
        fig = plt.figure()

    def reset(self, place_goal_nearby : bool, dist_threshold=4): 
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # 1 = wall, 2 = goal, 3 = agent
        num_walls = int(0.2 * self.grid_size**2)
        wall_indices = np.random.choice(self.grid_size**2, num_walls, replace=False)
        self.grid.flat[wall_indices] = 1
        self.visited = {}

        # Place goal
        while True:
            goal_pos = np.random.randint(0, self.grid_size, size=2)
            if self.grid[tuple(goal_pos)] == 0:
                self.goal_pos = tuple(goal_pos)
                self.grid[self.goal_pos] = 2
                break

        # Place agent
        while True:
            agent_pos = np.random.randint(0, self.grid_size, size=2)
            if self.grid[tuple(agent_pos)] == 0:
                if place_goal_nearby:
                    dist = np.sum(np.abs(self.goal_pos - agent_pos))
                    if dist > dist_threshold:
                        continue
                self.agent_pos = tuple(agent_pos)
                self.steps = 0
                break
        self.visited[self.agent_pos] = 1
        return self._get_obs()

    def _get_obs(self):
        obs = self.grid.copy()
        obs[self.agent_pos] = 3
        return obs

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        prev_dist = np.sum(np.abs(np.array(self.agent_pos) - np.array(self.goal_pos)))  # Manhattan

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = x + dx, y + dy
        
        illegal_move = False
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            if self.grid[nx, ny] != 1:
                self.agent_pos = (nx, ny)
                if (nx, ny) in self.visited:
                    self.visited[(nx, ny)] += 1
                else:
                    self.visited[(nx, ny)] = 1
            else:
                illegal_move = True  # wall
        else:
            illegal_move = True  # off-grid

        # Reward logic
        new_dist = np.sum(np.abs(np.array(self.agent_pos) - np.array(self.goal_pos)))
        delta_dist = prev_dist - new_dist

        reward = 0.0
        if self.agent_pos == self.goal_pos:
            reward = 10
        elif illegal_move:
            reward = -2
        else:
            reward = 0.5 * delta_dist - 0.05 # small positive for getting closer, minus small time penalty
            if self.visited[(self.agent_pos)] > 1:
                reward -= min(self.visited[(self.agent_pos)] * 0.05, 5)
            else:
                reward += 0.25
            
        

        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        return self._get_obs(), reward, done, {}


    def render(self, mode='human', episode_num=None):
        cmap = colors.ListedColormap(['white', 'black', 'green', 'blue'])
        plt.clf()  # clear previous frame
        plt.imshow(self._get_obs(), cmap=cmap, interpolation='nearest')
        plt.axis('off')
        if episode_num is not None:
            plt.title(f"Episode {episode_num}")
        plt.pause(0.2)  # pause to allow UI to update

    def close(self):
        plt.close()
