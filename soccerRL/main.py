from env.soccer_env import SoccerMultiAgentEnv
from agents.DQNAgent import DQNAgent
import wandb
import torch
import numpy as np

NUM_EPISODES = 500000
NUM_PLAYERS_PER_TEAM = 5
NUM_ACTIONS=9
MAX_STEPS=2000
SAVE_LAST_FREQ = 50
SAVE_ALL_FREQ = 5000
AGENT_UPDATE_FREQ = 20
device = torch.device('mps')



if __name__ == "__main__":
    wandb.init(
        project="gridworld-dqn",
        config={
            "Episodes": NUM_EPISODES,
            "Players per team" : NUM_PLAYERS_PER_TEAM,
            "Available actions": NUM_ACTIONS,
            "Max steps" : MAX_STEPS
        }
    )
    
    env = SoccerMultiAgentEnv(max_steps=MAX_STEPS)
    
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
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            step += 1
            #Sample actions from agents
            action_one = agent_one.act(obs=obs_tensor)
            action_two = agent_two.act(obs=obs_tensor)
            action = {"team_1" : action_one, "team_2" : action_two}
            
            #Advance game
            next_obs, rewards, terminated, truncated, info = env.step(action)
            next_obs_tensor = torch.tensor([next_obs], dtype=torch.float32, device=device)
            done = any(terminated.values()) or any(truncated.values())
            
            total_reward_one += rewards['team_1']
            total_reward_two += rewards['team_2']
            
            #Store info in replay buffers
            agent_one.store_transition(obs_tensor, action_one, rewards['team_1'], next_obs_tensor, done)
            agent_two.store_transition(obs_tensor, action_two, rewards['team_2'], next_obs_tensor, done)
            
            obs = next_obs
            
            
            if AGENT_UPDATE_FREQ % step == 0:
                #Update agents
                loss_one = agent_one.update()
                total_loss_one += float(loss_one) # type: ignore
                loss_two = agent_two.update()
                total_loss_two += float(loss_two) # type: ignore
            
            env.render() #Don't render during training
                        
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
            
        if rewards['team_1'] == 10:
            winner = 1
        elif rewards['team_2'] == 10:
            winner = 2
        else:
            winner = 0
        wandb.log({
            "Average Total Reward For Agent One Over Last 50 Episodes": np.mean(avg_reward_one),
            "Average Total Reward For Agent Two Over Last 50 Episodes": np.mean(avg_reward_two),
            "Average Total Steps Per Game for Last 50 Episodes": np.mean(avg_steps),
            "Steps": step,
            "Episode": episode,
            "Winner": winner,
            "Loss Per Step for Agent One": total_loss_one / step,
            "Loss Per Step for Agent Two": total_loss_two / step,
        })
        
        #Save models if applicable
        if episode % SAVE_LAST_FREQ == 0:
            save_path = 'models/last.pth'
            torch.save(agent_one.policy_net, 'models/lastAgentOne.pth')
            torch.save(agent_two.policy_net, 'models/lastAgentTwo.pth')
        if episode % SAVE_ALL_FREQ == 0:
            torch.save(agent_one.policy_net, f'models/AgentOne_{episode}.pth')
            torch.save(agent_two.policy_net, f'models/AgentTwo_{episode}.pth')
        
    env.close()
