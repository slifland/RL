from env.gridworld_env import GridWorldEnv
from agents.DQNAgent import DQNAgent
import torch
import wandb
import numpy as np

device = torch.device('mps')

def preprocess_obs(obs, action_buffer):
    """Consistent preprocessing that combines grid obs with action history"""
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).flatten()
    history_tensor = torch.tensor(action_buffer.action_buffer, dtype=torch.float32, device=device)
    return torch.cat([obs_tensor, history_tensor])

def one_hot_encode_grid(grid):
    """
    Converts a (H, W) grid to a (3, H, W) one-hot encoded tensor.
    """
    wall = (grid == 1).astype(np.float32)
    goal = (grid == 2).astype(np.float32)
    agent = (grid == 3).astype(np.float32)

    # Stack channels: shape (3, H, W)
    encoded = np.stack([wall, goal, agent], axis=0)
    return torch.tensor(encoded)  # shape: [3, H, W]

grid_size = 10
env = GridWorldEnv(grid_size=grid_size)
agent = DQNAgent(grid_w=grid_size, grid_h=grid_size)


num_episodes = 100000
max_steps_per_episode = env.max_steps
highest_reward_per_steps = float('-inf')


wandb.init(
    project="gridworld-dqn",
    config={
        "grid_size": grid_size,
        "buffer_size": agent.action_buffer.size,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon_start": agent.epsilon_start,
        "epsilon_end": agent.epsilon_end,
        "epsilon_decay": agent.epsilon_decay,
        "learning_rate": 1e-3,
        "update_freq": agent.update_freq,
        "num_episodes": num_episodes
    }
)   

#wandb.watch(agent.policy_net, log='all', log_freq=500)

constrain_goal_distance = True
constraint_drop=1000
constraint_distance = 2

for episode in range(num_episodes):
    if constrain_goal_distance and num_episodes > constraint_drop:
        constrain_goal_distance = False
    elif num_episodes > 400:
        constraint_distance = 3
        if num_episodes > 800:
            constraint_distance = 4
    obs = env.reset(constrain_goal_distance, dist_threshold=constraint_distance)
    #obs_tensor = preprocess_obs(obs, agent.action_buffer)
    obs_tensor = one_hot_encode_grid(obs)
    total_reward = 0
    total_loss = 0
    agent.reset_buffer()
    done = False

    for step in range(max_steps_per_episode):
        #env.render(episode_num=episode + 1)

        # Select action
        action, q_vals = agent.act(obs_tensor)
        if q_vals is not None:
            wandb.log({
                "q_max": q_vals.max().item(),
                "q_mean": q_vals.mean().item(),
                "q_min": q_vals.min().item(),
            })
        original_vec = torch.tensor(agent.action_buffer.action_buffer, dtype=torch.float32, device=device)
        agent.action_buffer.add_action(int(action))

        # Take step
        next_obs, reward, done, _ = env.step(action)
        # next_obs_tensor = preprocess_obs(next_obs, agent.action_buffer).unsqueeze(0)
        next_obs_tensor = one_hot_encode_grid(next_obs)

        # Store transition
        action_vec = torch.tensor(agent.action_buffer.action_buffer, dtype=torch.float32, device=device)
        agent.store_transition(obs_tensor, action, reward, next_obs_tensor, done, original_vec, action_vec)

        # Train the agent
        loss = agent.update()
        
        if loss is not None:
            wandb.log({"Loss":loss})

        obs_tensor = next_obs_tensor
        total_reward += reward
               
        if step > 0:
            reward_per_step = total_reward / step
        
        if done:
            break
 

    print(f"Episode {episode + 1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    wandb.log({
    "episode": episode + 1,
    "total_reward": total_reward,
    "epsilon": agent.epsilon,
    "reward_per_step":reward_per_step,
    })
    
    if (episode) % 50 == 0:
        checkpoint_path = f"models/DQNlast.pth"
        torch.save(agent.policy_net.state_dict(), checkpoint_path)
        print(f"✅ Saved model checkpoint to {checkpoint_path}")
    
    if reward_per_step > highest_reward_per_steps:
        checkpoint_path = f"models/best.pth"
        torch.save(agent.policy_net.state_dict(), checkpoint_path)
        print(f"✅ Saved best model to {checkpoint_path}")
        highest_reward_per_steps = reward_per_step
        

env.close()
wandb.finish()