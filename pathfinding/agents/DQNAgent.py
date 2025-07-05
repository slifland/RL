# === DQNAgent.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from .agent import Agent

device = torch.device('mps')

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, action_buffer, next_buffer):
        self.buffer.append((state, action, reward, next_state, done, action_buffer, next_buffer))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, buffers, next_buffers = zip(*batch)

        return (
            torch.stack(states),       # already torch tensors
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states),  # already torch tensors
            torch.tensor(dones, dtype=torch.float32, device=device),
            torch.stack(buffers),
            torch.stack(next_buffers)
        )


    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, obs):
        return self.fc(obs)
    
class CNNPolicy(nn.Module):
    def __init__(self, input_channels=3, grid_size=10, num_actions=4, buffer_size=10):
        super(CNNPolicy, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → down to 5x5
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # → down to 2x2
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, grid_size, grid_size)
            dummy_out = self.conv_layers(dummy_input)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size + buffer_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x, action_buffer):  # x: [B, C, H, W]; action_buffer: [B, buffer_size]            
            x = self.conv_layers(x)
            
            x = x.view(x.size(0), -1)  # flatten
            
            x = torch.cat([x, action_buffer], dim=1)  # concat action history
            
            x = self.fc_layers(x)
            return x

class DQNAgent(Agent):
    def __init__(self, grid_w, grid_h, epsilon=0.9, path=None, buffer_size=10):
        super().__init__(grid_w, grid_h, buffer_size)
        self.epsilon = epsilon
        self.device = device
        #self.input_dim =  grid_w * grid_h + buffer_size
        #self.policy_net = DQN(self.input_dim).to(self.device)
        #self.target_net = DQN(self.input_dim).to(self.device)
        self.policy_net = CNNPolicy(grid_size=grid_w).to(self.device)
        self.target_net = CNNPolicy(grid_size=grid_h).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if path:
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
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
            self.epsilon_decay = 50000

    def act(self, obs) -> tuple:
        action_vector = torch.tensor(self.action_buffer.action_buffer, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = obs.unsqueeze(0).to(device)
        move_set = [0, 1, 2, 3]
        if random.random() < self.epsilon:
            return random.choice(move_set), None
        with torch.no_grad():
            q_vals = self.policy_net(obs, action_vector)
            return torch.argmax(q_vals[0]).item(), q_vals

    def store_transition(self, obs, action, reward, next_obs, done, action_buffer, next_buffer):
        self.replay_buffer.push(obs, action, reward, next_obs, done, action_buffer, next_buffer)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Regular samples 
        regular_batch = self.replay_buffer.sample(self.batch_size)

        s, a, r, s2, d, action_buffers, next_buffers = regular_batch
        
        s = s.to(self.device)
        s2 = s2.to(self.device)
        action_buffers = action_buffers.to(self.device)
        next_buffers = next_buffers.to(self.device)
        q_vals = self.policy_net(s, action_buffers).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            best_actions = self.policy_net(s2, next_buffers).argmax(1)
            next_q_vals = self.target_net(s2, next_buffers).gather(1, best_actions.unsqueeze(1)).squeeze()
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

