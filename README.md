# RL

Playing around with OpenAI Gym framework. In pathfinding, agents learn to navigate a grid with walls to try to find a goal. In SoccerRL, two teams compete to get the ball into the opposing teams goal.

Modify the number of players and field size for soccer - if there are too few players for the field size, or the field is too big, the players will fail to learn. Adjust n_step in the DQN Agent as well for larger field sizes, because this will result in longer episodes.

Run main.py to train agents, and test.py to visualize the results.