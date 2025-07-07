# === DQNAgent.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from .agent import SoccerAgent
import sys

device = torch.device('mps')


class ReplayBuffer:
    def __init__(self, capacity=10000, num_players=5):
        self.buffer = deque(maxlen=capacity)
        self.num_players = num_players

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),  # shape [B, obs_dim]
            torch.tensor(np.array(actions), dtype=torch.int64, device=device).view(-1, self.num_players),  # [B, num_players]
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class SoccerPolicy(nn.Module):
    def __init__(self, num_players=5, num_actions=9):
        super().__init__()
        self.num_players = num_players
        self.num_actions = num_actions
        self.output_size = num_players * num_actions

        self.obs_dim = 4 * num_players * 2 + 4  # x, y, vx, vy for each agent on both teams, plus ball

        self.fc = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

    def forward(self, x):
        #print(x)
        x = self.fc(x)
        # for name, param in self.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"{name} contains NaNs")
        #     if torch.isinf(param).any():
        #         print(f"{name} contains Infs")
        return x.view(-1, self.num_players, self.num_actions)  # [B, num_players, num_actions]


class DQNAgent(SoccerAgent):
    def __init__(self, epsilon=0.9, path=None, num_players=5, num_actions=9, n_step=10):
        super().__init__(num_players=num_players)
        self.num_actions = num_actions
        self.num_players = num_players
        self.epsilon = epsilon
        self.device = device
        self.n_step = n_step
        self.gamma = 0.99

        self.policy_net = SoccerPolicy(num_players=num_players, num_actions=num_actions)
        self.target_net = SoccerPolicy(num_players=num_players, num_actions=num_actions)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if path:
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(num_players=num_players)
        self.goal_replay_buffer = ReplayBuffer(num_players=num_players, capacity=1000)
        self.n_step_buffer = deque(maxlen=n_step)

        self.batch_size = 128
        self.update_freq = 300
        self.steps = 0

        self.epsilon_start = epsilon
        self.epsilon_end = 0.05 if epsilon > 0 else 0
        self.epsilon_decay = 200000 if epsilon > 0 else 1

    def act(self, obs) -> tuple:
        obs = obs.unsqueeze(0) if obs.dim() == 1 else obs
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions, size=(self.num_players,)), None
        with torch.no_grad():
            q_vals = self.policy_net(obs)
            q_max = q_vals.max().item()
            q_min = q_vals.min().item()
            q_mean = q_vals.mean().item()
            argmax_actions = q_vals[0].argmax(dim=1)
            return tuple(argmax_actions.tolist()), (q_max, q_min, q_mean)

    def store_transition(self, obs, action, reward, next_obs, done):
        """Accumulate transitions and store n-step returns."""
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Compute n-step transition
        R = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * r

        obs_0, action_0, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_obs_n, done_n = self.n_step_buffer[-1]

        self.replay_buffer.push(obs_0, action_0, R, next_obs_n, done_n)
        if abs(R) > 15:
            self.goal_replay_buffer.push(obs_0, action_0, R, next_obs_n, done_n)

        if done:
            self.n_step_buffer.clear()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        goal_batch_size = min(int(self.batch_size * 0.05), len(self.goal_replay_buffer))
        regular_batch_size = self.batch_size - goal_batch_size

        if goal_batch_size > 0:
            s1, a1, r1, s2_1, d1 = self.goal_replay_buffer.sample(goal_batch_size)
            s2, a2, r2, s2_2, d2 = self.replay_buffer.sample(regular_batch_size)
            s = torch.cat([s1, s2], dim=0)
            a = torch.cat([a1, a2], dim=0)
            r = torch.cat([r1, r2], dim=0)
            s2 = torch.cat([s2_1, s2_2], dim=0)
            d = torch.cat([d1, d2], dim=0)
        else:
            s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)

        s, s2 = s.to(self.device), s2.to(self.device)
        q_vals_all = self.policy_net(s)
        q_vals = q_vals_all.gather(2, a.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            next_q_vals_all = self.target_net(s2)
            best_actions = self.policy_net(s2).argmax(dim=2)
            next_q_vals = next_q_vals_all.gather(2, best_actions.unsqueeze(2)).squeeze(2)
            target = r.unsqueeze(1) + (self.gamma ** self.n_step) * next_q_vals * (1 - d.unsqueeze(1))

        loss = F.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps / self.epsilon_decay)

        return loss.item()
