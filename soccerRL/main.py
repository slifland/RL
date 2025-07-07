from env.soccer_env import SoccerMultiAgentEnv
from agents.DQNAgent import DQNAgent
import wandb
import torch
import numpy as np

NUM_EPISODES = 500000
NUM_PLAYERS_PER_TEAM = 5
FIELD_HEIGHT = 800
FIELD_WIDTH = 1000
NUM_ACTIONS=9
MAX_STEPS=1000
SAVE_LAST_FREQ = 50
SAVE_ALL_FREQ = 20000
AGENT_UPDATE_FREQ = 20
device = torch.device('mps')



if __name__ == "__main__":
    wandb.init(
        project="RLSoccer",
        config={
            "Episodes": NUM_EPISODES,
            "Players per team" : NUM_PLAYERS_PER_TEAM,
            "Available actions": NUM_ACTIONS,
            "Max steps" : MAX_STEPS,
            "Field height" : FIELD_HEIGHT,
            "Field width" : FIELD_WIDTH
        }
    )
    
    env = SoccerMultiAgentEnv(max_steps=MAX_STEPS, num_players=NUM_PLAYERS_PER_TEAM, field_height=FIELD_HEIGHT, field_width=FIELD_WIDTH)
    
    agent_one = DQNAgent(num_players=NUM_PLAYERS_PER_TEAM, num_actions=NUM_ACTIONS)
    agent_two= DQNAgent(num_players=NUM_PLAYERS_PER_TEAM, num_actions=NUM_ACTIONS)
    
    #Following three tracks averages over last 50 episodes for a smoother graph
    avg_reward_one = []
    avg_reward_two = []
    avg_steps = []
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        step = 0
        total_reward_one = 0
        total_reward_two = 0
        total_loss_one = 0
        total_loss_two = 0
        while True:
            obs_tensor = torch.tensor(np.array(obs['team_1']), dtype=torch.float32, device=device).unsqueeze(0)
            step += 1
            #Sample actions from agents
            action_one, q_vals_one = agent_one.act(obs=obs_tensor)
            action_two, q_vals_two = agent_two.act(obs=obs_tensor)
            action = {"team_1" : action_one, "team_2" : action_two}
            
            #Advance game
            next_obs, rewards, terminated, truncated, info = env.step(action)
            next_obs_tensor = torch.tensor(np.array(obs['team_2']), dtype=torch.float32, device=device).unsqueeze(0)
            done = any(terminated.values()) or any(truncated.values())
            
            total_reward_one += rewards['team_1']
            total_reward_two += rewards['team_2']
                        
            #Store info in replay buffers
            agent_one.store_transition(obs_tensor, action_one, rewards['team_1'], next_obs_tensor, done)
            agent_two.store_transition(obs_tensor, action_two, rewards['team_2'], next_obs_tensor, done)
            obs = next_obs
            
            if step % AGENT_UPDATE_FREQ  == 0:
                loss_one = agent_one.update()
                if loss_one is not None:
                    total_loss_one += float(loss_one)

                loss_two = agent_two.update()
                if loss_two is not None:
                    total_loss_two += float(loss_two)
            
            # if q_vals_one:
            #     wandb.log({'Q_max_1': q_vals_one[0],
            #             'Q_min_1': q_vals_one[1],
            #             'Q_mean_1': q_vals_one[2]})
                    

                
            #env.render() #Don't render during training
                        
            #End if one team scores or hit max steps, and reset
            if done:
                break
            
        #End of episode logging
        if len(avg_reward_one) > 50:
            avg_reward_one = avg_reward_one[1:]
            avg_reward_two = avg_reward_two[1:]
            avg_steps = avg_steps[1:]
        avg_reward_one.append(total_reward_one)
        avg_reward_two.append(total_reward_two)
        avg_steps.append(step)
        
        if step == MAX_STEPS:
            winner = 0
        elif rewards['team_1'] > rewards['team_2']:
            winner = 1
        elif rewards['team_2'] > rewards['team_1']:
            winner = 2
        else:
            winner = 0
        wandb.log({
            "Average Total Reward For Agent One Over Last 50 Episodes": np.mean(avg_reward_one),
            "Average Total Reward For Agent Two Over Last 50 Episodes": np.mean(avg_reward_two),
            "Average Total Steps Per Game for Last 50 Episodes": np.mean(avg_steps),
            "Steps": step,
            "Agent One Epsilon": agent_one.epsilon,
        })
        match winner:
            case 1:
                win_str = "Team 1"
            case 0:
                win_str = "Timed Out"
            case 2:
                win_str = "Team 2"
        print(f"Episode: {episode}, Winner: {win_str}")
        
        #Save models if applicable
        if episode % SAVE_LAST_FREQ == 0:
            save_path = 'models/last.pth'
            torch.save(agent_one.policy_net.state_dict(), f'models/lastAgentOne_{NUM_PLAYERS_PER_TEAM}_players.pth')
            torch.save(agent_two.policy_net.state_dict(), f'models/lastAgentTwo_{NUM_PLAYERS_PER_TEAM}_players.pth')
        if episode % SAVE_ALL_FREQ == 0:
            torch.save(agent_one.policy_net.state_dict(), f'models/AgentOne_{episode}_{NUM_PLAYERS_PER_TEAM}_players.pth')
            torch.save(agent_two.policy_net.state_dict(), f'models/AgentTwo_{episode}_{NUM_PLAYERS_PER_TEAM}_players.pth')
        
    env.close()
