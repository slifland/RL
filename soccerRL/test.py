from env.soccer_env import SoccerMultiAgentEnv
from agents.DQNAgent import DQNAgent
import wandb
import torch
import numpy as np

NUM_EPISODES = 5
NUM_PLAYERS_PER_TEAM = 3
match NUM_PLAYERS_PER_TEAM:
    case 5:
        FIELD_HEIGHT = 800
        FIELD_WIDTH = 1200
    case 4:
        FIELD_HEIGHT = 500
        FIELD_WIDTH = 900
    case 1:
        FIELD_HEIGHT = 200
        FIELD_WIDTH = 600
    case 2:
        FIELD_HEIGHT = 300
        FIELD_WIDTH = 800
    case 3:
        FIELD_HEIGHT = 600
        FIELD_WIDTH = 1000
NUM_ACTIONS=9
MAX_STEPS=1000
device = torch.device('mps')



if __name__ == "__main__":
    
    env = SoccerMultiAgentEnv(max_steps=MAX_STEPS, num_players=NUM_PLAYERS_PER_TEAM, field_width = FIELD_WIDTH, field_height=FIELD_HEIGHT)
    
    agent_one = DQNAgent(num_players=NUM_PLAYERS_PER_TEAM, num_actions=NUM_ACTIONS, epsilon=0)
    agent_two= DQNAgent(num_players=NUM_PLAYERS_PER_TEAM, num_actions=NUM_ACTIONS, epsilon=0)
    
    agent_one.policy_net.load_state_dict(torch.load(f'models/lastAgentOne_{NUM_PLAYERS_PER_TEAM}_players.pth', map_location="mps"))
    agent_one.policy_net.eval()
    agent_two.policy_net.load_state_dict(torch.load(f'models/lastAgentTwo_{NUM_PLAYERS_PER_TEAM}_players.pth', map_location="mps"))
    agent_two.policy_net.eval()
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        while True:
            obs_tensor = torch.tensor(np.array(obs['team_1']), dtype=torch.float32, device=device).unsqueeze(0)
            #Sample actions from agents
            action_one, q_vals = agent_one.act(obs=obs_tensor)
            action_two, q_vals = agent_two.act(obs=obs_tensor)
            action = {"team_1" : action_one, "team_2" : action_two}
            
            #print(q_vals)
            
            #Advance game
            next_obs, rewards, terminated, truncated, info = env.step(action)
            done = any(terminated.values()) or any(truncated.values())
        
            obs = next_obs
            
            env.render()
                        
            #End if one team scores or hit max steps, and reset
            if done:
                break
            
        if rewards['team_1'] > rewards['team_2']:
            winner = 1
        elif rewards['team_2'] >  rewards['team_1']:
            winner = 2
        match winner:
            case 1:
                win_str = "Team 1"
            case 2:
                win_str = "Team 2"
        print(f"Episode: {episode}, Winner: {win_str}")
        
    
    env.close()
