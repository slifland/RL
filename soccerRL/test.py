from env.soccer_env import SoccerMultiAgentEnv
from agents.DQNAgent import DQNAgent
import wandb
import torch
import numpy as np

NUM_EPISODES = 5
NUM_PLAYERS_PER_TEAM = 5
NUM_ACTIONS=9
MAX_STEPS=2000
device = torch.device('mps')



if __name__ == "__main__":
    
    env = SoccerMultiAgentEnv(max_steps=MAX_STEPS)
    
    agent_one = DQNAgent(num_players=NUM_PLAYERS_PER_TEAM, num_actions=NUM_ACTIONS, epsilon=0)
    agent_two= DQNAgent(num_players=NUM_PLAYERS_PER_TEAM, num_actions=NUM_ACTIONS, epsilon=0)
    
    agent_one.policy_net.load_state_dict(torch.load('models/AgentOne_10000.pth', map_location="cpu"))
    agent_one.policy_net.eval()
    agent_two.policy_net.load_state_dict(torch.load('models/AgentTwo_10000.pth', map_location="cpu"))
    agent_two.policy_net.eval()
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        while True:
            obs_tensor = torch.tensor(np.array(obs['team_1']), dtype=torch.float32, device=device).unsqueeze(0)
            step += 1
            #Sample actions from agents
            action_one = agent_one.act(obs=obs_tensor)
            action_two = agent_two.act(obs=obs_tensor)
            action = {"team_1" : action_one, "team_2" : action_two}
            
            #Advance game
            next_obs, rewards, terminated, truncated, info = env.step(action)
            done = any(terminated.values()) or any(truncated.values())
        
            obs = next_obs
            
            env.render()
                        
            #End if one team scores or hit max steps, and reset
            if done:
                break
            
        if rewards['team_1'] == 10:
            winner = 1
        elif rewards['team_2'] == 10:
            winner = 2
        else:
            winner = 0
        match winner:
            case 1:
                win_str = "Team 1"
            case 0:
                win_str = "Timed Out"
            case 2:
                win_str = "Team 2"
        print(f"Episode: {episode}, Winner: {win_str}")
        
    
    env.close()
