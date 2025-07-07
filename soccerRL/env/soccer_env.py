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
MAX_VELOCITY = 300


class SoccerMultiAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, max_steps=2000, num_players = 5, field_width=400, field_height=200):
        self.num_players = num_players
        self.team_ids = ['team_1', 'team_2']
        self.agent_ids = self.team_ids
        self.field_width = field_width
        self.field_height = field_height
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
        goal_height = self.field_height * 2
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
                state.extend([body.position.x / self.field_width, body.position.y / self.field_height, body.velocity.x / MAX_VELOCITY, body.velocity.y / MAX_VELOCITY])
        state.extend([self.ball.position.x / self.field_width, self.ball.position.y / self.field_height, self.ball.velocity.x / MAX_VELOCITY, self.ball.velocity.y / MAX_VELOCITY])
        state = np.array(state, dtype=np.float32)
        return {team: state.copy() for team in self.team_ids}
    
    def _goal_progress_bonus(self, prev_ball_x, team):
        if team == "team_1":
            goal_x = self.field_width
        else:
            goal_x = 0
        new_ball_x = self.ball.position.x
        progress = (goal_x - new_ball_x) - (goal_x - prev_ball_x)
        return np.clip(progress / self.field_width, -1, 1) * 0.5
    
    def _ball_velocity_toward_goal_bonus(self):
        ball_vel = self.ball.velocity
        if ball_vel.length < 1e-5:
            return {"team_1": 0.0, "team_2": 0.0}

        dir_vector = ball_vel / ball_vel.length  # manually normalize to avoid NaNs
        team_1_dir = np.array([1, 0])
        team_2_dir = np.array([-1, 0])
        reward = {
            "team_1": float(np.dot(dir_vector, team_1_dir)) * 0.3,
            "team_2": float(np.dot(dir_vector, team_2_dir)) * 0.3,
        }
        return reward
    
    def _team_spread_bonus(self):
        bonus = {}
        for team in self.team_ids:
            positions = np.array([
                [self.agent_bodies[f"{team}_{i}"].position.x,
                self.agent_bodies[f"{team}_{i}"].position.y]
                for i in range(self.num_players)
            ], dtype=np.float32)

            if len(positions) < 2:
                bonus[team] = 0.0  # nothing to compute
                continue

            dists = [
                np.linalg.norm(positions[i] - positions[j])
                for i in range(len(positions)) for j in range(i + 1, len(positions))
            ]

            if len(dists) == 0:
                bonus[team] = 0.0
            else:
                mean_dist = np.mean(dists)
                bonus[team] = np.clip(mean_dist / self.field_width, 0, 1) * 0.1

        return bonus


    def step(self, action_dict):
        self.steps += 1
        for team in self.team_ids:
            for i in range(self.num_players):
                action_idx = action_dict[team][i]
                fx, fy = ACTION_DIRECTIONS[action_idx]
                self.agent_bodies[f"{team}_{i}"].apply_force_at_local_point((fx * 500, fy * 500))
        prev_ball_x = self.ball.position.x
        self.space.step(self.dt)

        for body in self.space.bodies:
            if body.velocity.length > MAX_VELOCITY:
                body.velocity = body.velocity.normalized() * MAX_VELOCITY

        truncated = self.steps > self.max_steps

        obs = self._get_obs()
        progress_bonus = {
            team: self._goal_progress_bonus(prev_ball_x, team)
            for team in self.team_ids
        }
        ball_velo_bonus = self._ball_velocity_toward_goal_bonus()
        spread_bonus = self._team_spread_bonus()
        rewards = self._get_rewards()
        infos = {team: {} for team in self.team_ids}
        # print("progress_bonus:", progress_bonus)
        # print("ball_velo_bonus:", ball_velo_bonus)
        # print("spread_bonus:", spread_bonus)
        # print("raw rewards:", rewards)
        for team in self.team_ids:
            rewards[team] += 3 * progress_bonus[team]
            rewards[team] += 3 * ball_velo_bonus[team]
            rewards[team] += 3 * spread_bonus[team]
            infos[team]["progress_bonus"] = 3 * progress_bonus[team]
            infos[team]["velocity_bonus"] = 3 * ball_velo_bonus[team]
            infos[team]["spread_bonus"] = 3 * spread_bonus[team]
        terminateds = self._get_terminated()
        truncateds = {team: truncated for team in self.team_ids}
        return obs, rewards, terminateds, truncateds, infos

    def _get_rewards(self):
        if self.goal_scored == "team_1":
            return {"team_1": 30.0, "team_2": -30.0}
        elif self.goal_scored == "team_2":
            return {"team_1": -30.0, "team_2": 30.0}
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
        goal_height = self.field_height * 0.8
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
        plt.title(str(self.steps))
        plt.pause(0.01)



    def close(self):
        if self.render_initialized:
            plt.ioff()
            plt.close(self.fig)
            self.render_initialized = False