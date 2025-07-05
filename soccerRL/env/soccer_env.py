import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import matplotlib.pyplot as plt


raw_dirs = np.array([
    [-1, -1], [0, -1], [1, -1],
    [-1,  0], [0,  0], [1,  0],
    [-1,  1], [0,  1], [1,  1],
], dtype=np.float32)

ACTION_DIRECTIONS = np.array([
    d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else d
    for d in raw_dirs
])


class SoccerMultiAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, max_steps=2000):
        self.num_players = 5
        self.team_ids = ['team_1', 'team_2']
        self.agent_ids = self.team_ids  # one agent per team
        self.field_width = 1000
        self.field_height = 600
        self.dt = 1 / 60.0
        
        self.max_steps = max_steps

        self.num_actions = len(ACTION_DIRECTIONS)

        self.action_space = {
            team: spaces.MultiDiscrete([self.num_actions] * self.num_players)
            for team in self.team_ids
        }

        obs_dim = 4 * (2 * self.num_players + 1)
        self.observation_space = {
            team: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for team in self.team_ids
        }

        self.render_mode = render_mode
        self._setup_space()

        self.fig, self.ax = plt.subplots()
        self.render_initialized = False

    def _setup_space(self):
        self.space = pymunk.Space()
        self.space.damping = 0.9
        self.space.gravity = (0, 0)
        self._create_boundaries()
        self._create_agents()
        self._create_ball()

    def _create_boundaries(self):
        static_lines = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.field_width, 0), 1),
            pymunk.Segment(self.space.static_body, (self.field_width, 0), (self.field_width, self.field_height), 1),
            pymunk.Segment(self.space.static_body, (self.field_width, self.field_height), (0, self.field_height), 1),
            pymunk.Segment(self.space.static_body, (0, self.field_height), (0, 0), 1),
        ]
        for line in static_lines:
            line.elasticity = 1.0
        self.space.add(*static_lines)

    def _create_circle_body(self, pos, radius=15, mass=1):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 1.0
        self.space.add(body, shape)
        return body

    def _create_agents(self):
        self.agent_bodies = {}
        spacing = self.field_height / (self.num_players + 1)
        x_margin = self.field_width * 0.1

        for i in range(self.num_players):
            y = spacing * (i + 1)
            self.agent_bodies[f"team_1_{i}"] = self._create_circle_body((x_margin, y))
            self.agent_bodies[f"team_2_{i}"] = self._create_circle_body((self.field_width - x_margin, y))

    def _create_ball(self):
        self.ball = self._create_circle_body((self.field_width / 2, self.field_height / 2), radius=10, mass=0.5)

    def reset(self, *, seed=None, options=None):
        self._setup_space()
        obs = self._get_obs()
        infos = {team: {} for team in self.team_ids}
        self.steps = 0
        return obs, infos

    def _get_obs(self):
        state = []
        for i in range(self.num_players):
            for team in self.team_ids:
                body = self.agent_bodies[f"{team}_{i}"]
                state.extend([body.position.x, body.position.y, body.velocity.x, body.velocity.y])
        state.extend([self.ball.position.x, self.ball.position.y, self.ball.velocity.x, self.ball.velocity.y])
        state = np.array(state, dtype=np.float32)
        return {team: state.copy() for team in self.team_ids}

    def step(self, action_dict):
        self.steps += 1
        for team in self.team_ids:
            for i in range(self.num_players):
                action_idx = action_dict[team][i]
                fx, fy = ACTION_DIRECTIONS[action_idx]
                self.agent_bodies[f"{team}_{i}"].apply_force_at_local_point((fx * 500, fy * 500))

        self.space.step(self.dt)
        
        truncated = self.steps > self.max_steps

        obs = self._get_obs()
        rewards = self._get_rewards()
        terminateds = self._get_terminated()
        truncateds = {team: truncated for team in self.team_ids}
        infos = {team: {} for team in self.team_ids}

        return obs, rewards, terminateds, truncateds, infos

    def _get_rewards(self):
        ball_x = self.ball.position.x
        if ball_x < 0:
            return {"team_1": -10.0, "team_2": 10.0}
        elif ball_x > self.field_width:
            return {"team_1": 10.0, "team_2": -10.0}
        else:
            halfway = self.field_width / 2
            ball_dist_x = abs(halfway - ball_x)
            bonus = ball_dist_x / halfway #reward for ball being further on opponent side
            if ball_x > halfway: 
                return {"team_1" : bonus, "team_2" : -bonus}
            else:
                return {"team_1" : -bonus, "team_2" : bonus}

    def _get_terminated(self):
        ball_x = self.ball.position.x
        done = ball_x < 0 or ball_x > self.field_width
        return {team: done for team in self.team_ids}

    def render(self, mode='human'):
        if not self.render_initialized:
            plt.ion()
            self.render_initialized = True

        self.ax.clear()
        self.ax.set_xlim(0, self.field_width)
        self.ax.set_ylim(0, self.field_height)
        self.ax.set_aspect('equal')
        self.ax.set_title("2D Soccer - MultiAgent")

        self.ax.plot([0, self.field_width], [0, 0], 'k-')
        self.ax.plot([0, self.field_width], [self.field_height, self.field_height], 'k-')
        self.ax.plot([0, 0], [0, self.field_height], 'g-', linewidth=3)
        self.ax.plot([self.field_width, self.field_width], [0, self.field_height], 'r-', linewidth=3)

        for agent, body in self.agent_bodies.items():
            color = 'blue' if 'team_1' in agent else 'red'
            circle = plt.Circle((body.position.x, body.position.y), 15, color=color, alpha=0.8)
            self.ax.add_patch(circle)

        ball = plt.Circle((self.ball.position.x, self.ball.position.y), 10, color='black')
        self.ax.add_patch(ball)

        plt.pause(0.01)

    def close(self):
        if self.render_initialized:
            plt.ioff()
            plt.close(self.fig)
            self.render_initialized = False
