# === DQNAgent.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from .agent import SoccerAgent

device = torch.device('mps')

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),       # already torch tensors
            torch.tensor(np.array(actions), dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states),  # already torch tensors
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
        
        self.obs_dim = 4 * num_players * 2 + 4 #x, y, vx, vy for each agent on both teams, plus x, y, vx, vy for ball
        
        self.fc = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )
        
    def forward(self, x):
        x = self.fc(x)
        #Returns shape [B, num_players, num_actions]
        return x.view(-1, self.num_players, self.num_actions)
        

class DQNAgent(SoccerAgent):
    def __init__(self, epsilon=0.9, path=None, num_players=5, num_actions=9):
        super().__init__(num_players=num_players)
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.device = device
        #self.input_dim =  grid_w * grid_h + buffer_size
        #self.policy_net = DQN(self.input_dim).to(self.device)
        #self.target_net = DQN(self.input_dim).to(self.device)
        self.policy_net = SoccerPolicy(num_players=num_players, num_actions=num_actions).to(self.device)
        self.target_net = SoccerPolicy(num_players=num_players, num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if path:
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=5e-5)
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.99
        self.steps = 0
        self.update_freq = 1000
        
        self.epsilon_start = epsilon
        if epsilon == 0:
            self.epsilon_end = 0
            self.epsilon_decay = 1
        else:
            self.epsilon_end = 0.05
            self.epsilon_decay = 100000

    def act(self, obs) -> tuple:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 9, size=(self.num_players,)), None
        else:
            q_vals = self.policy_net(obs)
            q_max = q_vals.max().item()
            q_min = q_vals.min().item()
            q_mean = q_vals.mean().item()
            argmax_actions = q_vals[0].argmax(dim=1) #shape: [num_players]
            action_tuple = tuple(argmax_actions.tolist())
            return action_tuple, (q_max, q_min, q_mean)

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Regular samples 
        regular_batch = self.replay_buffer.sample(self.batch_size)

        s, a, r, s2, d = regular_batch
        
        s = s.to(self.device)
        s2 = s2.to(self.device)
        q_vals = self.policy_net(s).gather(2, a.unsqueeze(2)).squeeze(2) #shape: [B, num_actions]
        with torch.no_grad():
            best_actions = self.policy_net(s2).argmax(dim=2) #shape: [B, num_actions]
            next_q_vals = self.target_net(s2).gather(2, best_actions.unsqueeze(2)).squeeze(2) #shape: [B, num_actions]
            r = r.unsqueeze(1)
            d = d.unsqueeze(1)
            target = r + self.gamma * next_q_vals * (1 - d)
            target = target.clamp(-10, 10)

        loss = F.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1

        if self.steps % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > 0:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                np.exp(-1. * self.steps / self.epsilon_decay)
        
        return loss.item()

