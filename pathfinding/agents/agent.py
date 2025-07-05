# === agent.py ===
import random

class ActionBuffer:
    def __init__(self, size: int = 10):
        self.size = size
        self.action_buffer = [-1] * size

    def add_action(self, action: int | None):
        self.action_buffer = self.action_buffer[1:]
        self.action_buffer.append(action if action is not None else -1)

    def values(self):
        return self.action_buffer
    
    def reset(self):
        self.action_buffer = [-1] * self.size

class Agent:
    def __init__(self, grid_w, grid_h, buffer_size=10):
        self.w = grid_w
        self.h = grid_h
        self.action_buffer = ActionBuffer(buffer_size)
        
    def reset_buffer(self):
        self.action_buffer.reset()

    def act(self, obs):
        return random.choice([0, 1, 2, 3])

