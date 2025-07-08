import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.fc(x)  # [B, num_actions]


class DQNAgent:
    def __init__(self, obs_dim, num_actions=9, epsilon=1.0, n_step=3, gamma=0.99):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.device = device

        self.policy_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.update_freq = 200
        self.steps = 0

        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = 0.05
        self.epsilon_decay = 10000

        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    def act(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_vals = self.policy_net(obs)
            return q_vals.argmax(dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        obs = torch.tensor(obs, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # n-step return
        R = sum((self.gamma ** i) * t[2] for i, t in enumerate(self.n_step_buffer))
        obs_0, action_0, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_obs_n, done_n = self.n_step_buffer[-1]

        self.replay_buffer.push(obs_0, action_0, R, next_obs_n, done_n)

        if done:
            self.n_step_buffer.clear()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        s , a, r, s2, d = self.replay_buffer.sample(self.batch_size)

        q_vals = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.target_net(s2).max(dim=1)[0]
            target = r + (self.gamma ** self.n_step) * next_q_vals * (1 - d)

        loss = F.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps / self.epsilon_decay)
        )

        return loss.item()
