import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import Sequence

class ActorCritic(nn.Module):
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(size)(x))
        logits = nn.Dense(1)(x)
        value = nn.Dense(1)(x)
        return logits.squeeze(-1), value.squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), lr=1e-4, gamma=0.99, clip_eps=0.2):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.model = ActorCritic(hidden_sizes)
        dummy_input = jnp.zeros((obs_dim,))
        self.params = self.model.init(jax.random.PRNGKey(0), dummy_input)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

    def select_action(self, obs, rng):
        logits, _ = self.model.apply(self.params, jnp.array(obs))
        noise = jax.random.normal(rng, shape=logits.shape)
        action = jnp.clip(logits + 0.1 * noise, 0.0, 1.0)
        return action

    def compute_advantages(self, rewards, values, next_values, dones):

        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        def scan_fn(carry, delta):
            adv = delta + self.gamma * carry
            return adv, adv

        _, advantages = jax.lax.scan(
            f=scan_fn,
            init=0.0,
            xs=deltas[::-1]
        )
        return advantages[::-1]  # Inverter para a ordem original

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch

        def loss_fn(params):
            logits, values = jax.vmap(lambda x: self.model.apply(params, x))(obs)
            _, next_values = jax.vmap(lambda x: self.model.apply(params, x))(next_obs)

            advantages = self.compute_advantages(rewards, values, next_values, dones)
            returns = advantages + values

            ratio = (logits - actions) / (1e-8 + jnp.std(actions))
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = jnp.mean((returns - values) ** 2)

            return policy_loss + 0.5 * value_loss

        grads = jax.grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
