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

MAX_VELOCITY = 300


class SumoMultiAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, max_steps=2000, num_agents=6, width=400, height=400):
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.width = width
        self.height = height
        self.dt = 1 / 30
        self.max_steps = max_steps
        self.num_actions = len(ACTION_DIRECTIONS)
        self.render_mode = render_mode

        self.left = 20
        self.right = self.width - 20
        self.bottom = 20
        self.top = self.height - 20

        self.action_space = {
            agent: spaces.Discrete(self.num_actions)
            for agent in self.agent_ids
        }

        obs_dim = 4 * self.num_agents + 1  # all agents observe everyone + own dist to edge
        self.observation_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agent_ids
        }

        self._setup_space()

        self.fig, self.ax = plt.subplots()
        self.render_initialized = False

    def _setup_space(self):
        self.space = pymunk.Space()
        self.space.damping = 0.9
        self.space.gravity = (0, 0)
        self._create_agents()

    def _create_circle_body(self, pos, radius=15, mass=1):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.damping = 0.9
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 1.0
        self.space.add(body, shape)
        return body

    def _create_agents(self):
        self.agent_bodies = {}
        angle_step = 2 * np.pi / self.num_agents
        spawn_radius = min(self.right - self.left, self.top - self.bottom) * 0.3
        center = np.array([self.width / 2, self.height / 2])

        for i, agent in enumerate(self.agent_ids):
            angle = i * angle_step
            x = center[0] + spawn_radius * np.cos(angle)
            y = center[1] + spawn_radius * np.sin(angle)
            self.agent_bodies[agent] = self._create_circle_body((x, y))

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.eliminated = {agent: False for agent in self.agent_ids}

        angle_step = 2 * np.pi / self.num_agents
        spawn_radius = min(self.right - self.left, self.top - self.bottom) * 0.3
        center = np.array([self.width / 2, self.height / 2])

        for i, agent in enumerate(self.agent_ids):
            angle = i * angle_step
            x = center[0] + spawn_radius * np.cos(angle)
            y = center[1] + spawn_radius * np.sin(angle)
            self._reset_body(self.agent_bodies[agent], (x, y))

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agent_ids}
        return obs, infos

    def _reset_body(self, body, position):
        body.position = position
        body.velocity = (0, 0)
        body.angle = 0
        body.angular_velocity = 0

    def _get_obs(self):
        obs = {}
        state = []
        for agent in self.agent_ids:
            body = self.agent_bodies[agent]
            state.extend([
                body.position.x / self.width,
                body.position.y / self.height,
                body.velocity.x / MAX_VELOCITY,
                body.velocity.y / MAX_VELOCITY,
            ])

        for agent in self.agent_ids:
            body = self.agent_bodies[agent]
            dx = min(body.position.x - self.left, self.right - body.position.x)
            dy = min(body.position.y - self.bottom, self.top - body.position.y)
            dist_to_edge = min(dx, dy) / (self.width / 2)

            full_obs = state.copy()
            full_obs.append(dist_to_edge)
            obs[agent] = np.array(full_obs, dtype=np.float32)

        return obs

    def step(self, action_dict):
        self.steps += 1

        for agent in self.agent_ids:
            if self.eliminated[agent]:
                continue
            action_idx = action_dict[agent]
            fx, fy = ACTION_DIRECTIONS[action_idx]
            self.agent_bodies[agent].apply_force_at_local_point((fx * 50, fy * 50))

        self.space.step(self.dt)

        for body in self.space.bodies:
            if body.velocity.length > MAX_VELOCITY:
                body.velocity = body.velocity.normalized() * MAX_VELOCITY

        obs = self._get_obs()
        rewards = self._get_rewards()
        terminateds = self._get_terminated()
        truncated = self.steps > self.max_steps
        truncateds = {agent: truncated for agent in self.agent_ids}
        infos = {agent: {} for agent in self.agent_ids}
        return obs, rewards, terminateds, truncateds, infos

    def _get_rewards(self):
        rewards = {agent: 0.0 for agent in self.agent_ids}
        for agent in self.agent_ids:
            if self.eliminated[agent]:
                continue
            x, y = self.agent_bodies[agent].position
            if not (self.left <= x <= self.right and self.bottom <= y <= self.top):
                rewards[agent] -= 10.0
            else:
                for other in self.agent_ids:
                    if other == agent or self.eliminated[other]:
                        continue
                    ox, oy = self.agent_bodies[other].position
                    if not (self.left <= ox <= self.right and self.bottom <= oy <= self.top):
                        rewards[agent] += 5.0
        return rewards

    def _get_terminated(self):
        terminated = {}
        for agent in self.agent_ids:
            if self.eliminated[agent]:
                terminated[agent] = True
                continue
            x, y = self.agent_bodies[agent].position
            if not (self.left <= x <= self.right and self.bottom <= y <= self.top):
                self.eliminated[agent] = True
                terminated[agent] = True
            else:
                terminated[agent] = False
        return terminated

    def render(self, mode='human'):
        if not self.render_initialized:
            plt.ion()
            self.render_initialized = True

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_title("Sumo Square - MultiAgent")

        # Draw square boundary
        square = plt.Rectangle((self.left, self.bottom), self.right - self.left, self.top - self.bottom,
                               fill=False, edgecolor='black', linewidth=2)
        self.ax.add_patch(square)

        for agent, body in self.agent_bodies.items():
            color = 'gray' if self.eliminated[agent] else 'blue'
            circle = plt.Circle((body.position.x, body.position.y), 15, color=color, alpha=0.8)
            self.ax.add_patch(circle)

        plt.title(f"Step: {self.steps}")
        plt.pause(0.01)

    def close(self):
        if self.render_initialized:
            plt.ioff()
            plt.close(self.fig)
            self.render_initialized = False
