from env.sumo_env import SumoMultiAgentEnv  # or soccer_env if using teams
from agents.DQNAgent import DQNAgent
import torch
import numpy as np
import os

# === Config ===
NUM_EPISODES = 5
NUM_AGENTS = 5
NUM_ACTIONS = 9
MAX_STEPS = 700
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Optional: adjust field size if needed
FIELD_SIZES = {
    1: (100, 100),
    2: (200, 200),
    3: (300, 300),
    4: (400, 400),
    5: (500, 500),
}
FIELD_WIDTH, FIELD_HEIGHT = FIELD_SIZES.get(NUM_AGENTS, (1000, 600))

if __name__ == "__main__":
    env = SumoMultiAgentEnv(
        max_steps=MAX_STEPS,
        num_agents=NUM_AGENTS,
        width=FIELD_WIDTH,
        height=FIELD_HEIGHT
    )

    agent_ids = env.agent_ids
    obs_dim = env.observation_space[agent_ids[0]].shape[0]

    # === Load one DQNAgent per agent_id ===
    agents = {}
    for agent_id in agent_ids:
        agent = DQNAgent(obs_dim=obs_dim, num_actions=NUM_ACTIONS, epsilon=0)
        model_path = f"models/last_{agent_id}_{NUM_AGENTS}_agents.pth"
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval()
        agents[agent_id] = agent
        print(f"Loaded model for {agent_id} from {model_path}")

    # === Run evaluation episodes ===
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {
                agent_id: agents[agent_id].act(obs[agent_id])
                for agent_id in agent_ids
            }

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = sum(not terminated[aid] for aid in agent_ids) <= 1 or any(truncated.values())

            obs = next_obs
            env.render()

        # === Determine winner ===
        alive = [aid for aid in agent_ids if not terminated[aid]]
        if len(alive) == 1:
            print(f"Episode {episode}: Winner is {alive[0]}")
        else:
            print(f"Episode {episode}: Draw or timeout")

    env.close()
