# === agent.py ===
import random
import numpy as np

class SoccerAgent:
    def __init__(self, num_players):
        self.num_players=num_players
        
    def act(self, obs):
        return np.random.randint(0, 9, size=(self.num_players,))

