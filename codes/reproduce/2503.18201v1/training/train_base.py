# training/train_base.py

import jax
import jax.numpy as jnp

import numpy as np
from envs.supply_chain_env import SupplyChainEnv
from models.ppo_agent import PPOAgent
from training.utils import set_seed


def train_base_model(scenario: str):
    config_path = f"configs/scenario_{scenario}.yaml"
    env = SupplyChainEnv(config_path)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)

    num_episodes = 50
    max_steps = 128
    rng = jax.random.PRNGKey(0)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0

        obs_buffer = []
        act_buffer = []
        rew_buffer = []
        next_obs_buffer = []
        done_buffer = []

        for step in range(max_steps):
            rng, subkey = jax.random.split(rng)
            action = agent.select_action(obs, subkey)
            action_np = np.array(action)

            next_obs, reward, done, _, _ = env.step(action_np)

            obs_buffer.append(obs)
            act_buffer.append(action_np)
            rew_buffer.append(reward)
            next_obs_buffer.append(next_obs)
            done_buffer.append(done)

            ep_reward += reward
            obs = next_obs

            if done:
                break

        batch = (
            jnp.array(obs_buffer),
            jnp.array(act_buffer),
            jnp.array(rew_buffer),
            jnp.array(next_obs_buffer),
            jnp.array(done_buffer, dtype=jnp.float32),
        )

        agent.update(batch)
        print(f"Episode {episode+1} - Reward: {ep_reward:.2f}")
