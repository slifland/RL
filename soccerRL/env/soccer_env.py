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

BALL_COLLISION_TYPE = 1
GOAL_SENSOR_TYPE = 2

class SoccerMultiAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, max_steps=2000):
        self.num_players = 5
        self.team_ids = ['team_1', 'team_2']
        self.agent_ids = self.team_ids
        self.field_width = 1000
        self.field_height = 600
        self.dt = 1 / 30

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
        self.goal_scored = None
        self._create_boundaries()
        self._create_agents()
        self._create_ball()
        self._create_goal_sensors()

        self.space.on_collision(
        BALL_COLLISION_TYPE,
        GOAL_SENSOR_TYPE,
        begin=self._handle_goal_collision
         )

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

    def _create_goal_sensors(self):
        goal_height = self.field_height * 0.4
        goal_y_min = (self.field_height - goal_height) / 2
        goal_y_max = goal_y_min + goal_height

        # Slightly inside the field, just before the wall at x=0 and x=field_width
        left_goal = pymunk.Segment(self.space.static_body, (0.5, goal_y_min), (0.5, goal_y_max), 1)
        left_goal.sensor = True
        left_goal.collision_type = GOAL_SENSOR_TYPE

        right_goal = pymunk.Segment(self.space.static_body, (self.field_width - 0.5, goal_y_min), (self.field_width - 0.5, goal_y_max), 1)
        right_goal.sensor = True
        right_goal.collision_type = GOAL_SENSOR_TYPE

        self.space.add(left_goal, right_goal)


    def _handle_goal_collision(self, arbiter, space, data):
        contact_point = arbiter.contact_point_set.points[0].point_a
        if contact_point.x < self.field_width / 2:
            self.goal_scored = "team_2"
        else:
            self.goal_scored = "team_1"
        return False

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
        self.ball = self._create_circle_body((self.field_width / 2, self.field_height / 2), radius=10, mass=0.25)
        for shape in self.ball.shapes:
            shape.collision_type = BALL_COLLISION_TYPE

    def reset(self, *, seed=None, options=None):
        self.goal_scored = None
        self.steps = 0

        # Reset agent positions
        spacing = self.field_height / (self.num_players + 1)
        x_margin = self.field_width * 0.1
        for i in range(self.num_players):
            y = spacing * (i + 1)
            self._reset_body(self.agent_bodies[f"team_1_{i}"], (x_margin, y))
            self._reset_body(self.agent_bodies[f"team_2_{i}"], (self.field_width - x_margin, y))

        # Reset ball to center
        self._reset_body(self.ball, (self.field_width / 2, self.field_height / 2))

        obs = self._get_obs()
        infos = {team: {} for team in self.team_ids}
        return obs, infos
    
    def _reset_body(self, body, position):
        body.position = position
        body.velocity = (0, 0)
        body.angle = 0
        body.angular_velocity = 0

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
        MAX_VELOCITY = 300

        for body in self.space.bodies:
            if body.velocity.length > MAX_VELOCITY:
                body.velocity = body.velocity.normalized() * MAX_VELOCITY

        truncated = self.steps > self.max_steps

        obs = self._get_obs()
        rewards = self._get_rewards()
        terminateds = self._get_terminated()
        truncateds = {team: truncated for team in self.team_ids}
        infos = {team: {} for team in self.team_ids}

        return obs, rewards, terminateds, truncateds, infos

    def _get_rewards(self):
        if self.goal_scored == "team_1":
            return {"team_1": 10.0, "team_2": -10.0}
        elif self.goal_scored == "team_2":
            return {"team_1": -10.0, "team_2": 10.0}
        else:
            ball_x = self.ball.position.x
            halfway = self.field_width / 2
            progress = (ball_x - halfway) / halfway  # ranges from -1 to 1
            return {
                "team_1": progress,
                "team_2": -progress
            }

    def _get_terminated(self):
        done = self.goal_scored is not None
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

        # Draw field boundaries
        self.ax.plot([0, self.field_width], [0, 0], 'k-')  # bottom
        self.ax.plot([0, self.field_width], [self.field_height, self.field_height], 'k-')  # top
        self.ax.plot([0, 0], [0, self.field_height], 'k-')  # left
        self.ax.plot([self.field_width, self.field_width], [0, self.field_height], 'k-')  # right

        # Goal areas (drawn just inside the field)
        goal_height = self.field_height * 0.4
        goal_y_min = (self.field_height - goal_height) / 2

        # Left goal: green shaded box inside the field
        self.ax.add_patch(
            plt.Rectangle((0, goal_y_min), 10, goal_height, color='green', alpha=0.4, label="Team 2 Goal")
        )

        # Right goal: red shaded box inside the field
        self.ax.add_patch(
            plt.Rectangle((self.field_width - 10, goal_y_min), 10, goal_height, color='red', alpha=0.4, label="Team 1 Goal")
        )

        # Draw agents
        for agent, body in self.agent_bodies.items():
            color = 'blue' if 'team_1' in agent else 'red'
            circle = plt.Circle((body.position.x, body.position.y), 15, color=color, alpha=0.8)
            self.ax.add_patch(circle)

        # Draw ball
        ball = plt.Circle((self.ball.position.x, self.ball.position.y), 10, color='black')
        self.ax.add_patch(ball)

        # Optional: add center line
        self.ax.plot([self.field_width / 2, self.field_width / 2], [0, self.field_height], 'gray', linestyle='--', alpha=0.5)

        plt.pause(0.01)



    def close(self):
        if self.render_initialized:
            plt.ioff()
            plt.close(self.fig)
            self.render_initialized = False