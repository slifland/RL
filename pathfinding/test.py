from env.gridworld_env import GridWorldEnv
from agents.DQNAgent import DQNAgent
import torch
import numpy as np

device = torch.device('mps')

def one_hot_encode_grid(grid):
    """
    Converts a (H, W) grid to a (3, H, W) one-hot encoded tensor.
    """
    wall = (grid == 1).astype(np.float32)
    goal = (grid == 2).astype(np.float32)
    agent = (grid == 3).astype(np.float32)
    encoded = np.stack([wall, goal, agent], axis=0)  # shape: (3, H, W)
    return torch.tensor(encoded, dtype=torch.float32, device=device)

env = GridWorldEnv(grid_size=10)
agent = DQNAgent(grid_w=10, grid_h=10, epsilon=0)

# Load trained model
agent.policy_net.load_state_dict(torch.load('models/DQNlast.pth'))
agent.policy_net.eval()

num_episodes = 10
max_steps_per_episode = env.max_steps

for episode in range(num_episodes):
    obs = env.reset(False)
    obs_tensor = one_hot_encode_grid(obs).unsqueeze(0)  # shape: [1, 3, H, W]
    agent.reset_buffer()
    total_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        env.render(episode_num=episode + 1)

        # Prepare action buffer
        action_buffer_tensor = torch.tensor(agent.action_buffer.action_buffer, dtype=torch.float32, device=device).unsqueeze(0)

        # Inference
        with torch.no_grad():
            q_vals = agent.policy_net(obs_tensor, action_buffer_tensor)
            action = torch.argmax(q_vals, dim=1).item()

        # Step
        agent.action_buffer.add_action(action)
        next_obs, reward, done, _ = env.step(action)
        obs_tensor = one_hot_encode_grid(next_obs).unsqueeze(0)  # update obs
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1} | Reward: {total_reward:.2f}")

env.close()
