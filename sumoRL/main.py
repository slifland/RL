from env.sumo_env import SumoMultiAgentEnv
from agents.DQNAgent import DQNAgent
import wandb
import torch
import numpy as np
import os

NUM_EPISODES = 500_000
NUM_AGENTS = 5
FIELD_SIZES = {
    1: (100, 100),
    2: (200, 200),
    3: (300, 300),
    4: (400, 400),
    5: (500, 500),
}
FIELD_WIDTH, FIELD_HEIGHT = FIELD_SIZES.get(NUM_AGENTS, (600, 600))
NUM_ACTIONS = 9
MAX_STEPS = 700
SAVE_LAST_FREQ = 50
SAVE_ALL_FREQ = 20_000
AGENT_UPDATE_FREQ = 2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    wandb.init(
        project="RLSumo",
        config={
            "Episodes": NUM_EPISODES,
            "Agents": NUM_AGENTS,
            "Available actions": NUM_ACTIONS,
            "Max steps": MAX_STEPS,
        }
    )

    env = SumoMultiAgentEnv(max_steps=MAX_STEPS, num_agents=NUM_AGENTS, width = FIELD_WIDTH, height = FIELD_HEIGHT)
    agent_ids = env.agent_ids
    obs_dim = 4 * NUM_AGENTS + 1 #position, velocity for each agent, personal distance from center

    # Create one DQNAgent per environment agent
    agents = {
        agent_id: DQNAgent(obs_dim=obs_dim, num_actions=NUM_ACTIONS)
        for agent_id in agent_ids
    }

    avg_rewards = {agent_id: [] for agent_id in agent_ids}
    avg_steps = []

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        total_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        step = 0
        terminated = {agent_id: False for agent_id in agent_ids}
        truncated = {agent_id: False for agent_id in agent_ids}

        while True:
            actions = {
                agent_id: agents[agent_id].act(obs[agent_id])
                for agent_id in agent_ids
                if not terminated[agent_id]
            }

            next_obs, rewards, terminated, truncated, info = env.step(actions)

            active_agents = sum(not terminated[agent_id] for agent_id in agent_ids)
            done = active_agents <= 1 or any(truncated.values())

            # Store transitions and accumulate rewards
            for agent_id in actions:  # only those that acted
                agents[agent_id].store_transition(
                    obs[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_obs[agent_id],
                    terminated[agent_id] or truncated[agent_id]
                )
                total_rewards[agent_id] += rewards[agent_id]

            obs = next_obs
            step += 1

            if step % AGENT_UPDATE_FREQ == 0:
                for agent_id in agent_ids:
                    if not terminated[agent_id]:
                        agents[agent_id].update()

            if done:
                break

        # Track average reward and steps
        avg_steps.append(step)
        for agent_id in agent_ids:
            avg_rewards[agent_id].append(total_rewards[agent_id])
            if len(avg_rewards[agent_id]) > 50:
                avg_rewards[agent_id] = avg_rewards[agent_id][1:]
        if len(avg_steps) > 50:
            avg_steps = avg_steps[1:]

        # Logging
        log_data = {
            "Average Total Steps (Last 50)": np.mean(avg_steps),
            "Steps": step,
        }
        for agent_id in agent_ids:
            log_data[f"Avg Reward {agent_id} (Last 50)"] = np.mean(avg_rewards[agent_id])
            log_data[f"Epsilon {agent_id}"] = agents[agent_id].epsilon
        wandb.log(log_data)

        print(f"Episode {episode} | Steps: {step} | Avg Reward: {[round(np.mean(avg_rewards[aid]), 2) for aid in agent_ids]}")

        # Save models
        os.makedirs("models", exist_ok=True)
        if episode % SAVE_LAST_FREQ == 0:
            for agent_id in agent_ids:
                torch.save(
                    agents[agent_id].policy_net.state_dict(),
                    f"models/last_{agent_id}.pth"
                )
        if episode % SAVE_ALL_FREQ == 0:
            for agent_id in agent_ids:
                torch.save(
                    agents[agent_id].policy_net.state_dict(),
                    f"models/{agent_id}_ep{episode}.pth"
                )

    env.close()
