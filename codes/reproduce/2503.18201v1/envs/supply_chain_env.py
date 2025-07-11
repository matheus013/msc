# Simulação do ambiente MEIO (simplificado)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml

class SupplyChainEnv(gym.Env):
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.num_stock_points = 4
        self.max_inventory = 100
        self.max_order = 50
        self.action_space = spaces.Box(low=0, high=self.max_order, shape=(self.num_stock_points,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.max_inventory, high=self.max_inventory, shape=(self.num_stock_points,), dtype=np.float32)
        self.holding_cost = np.array([1.0] * self.num_stock_points)
        self.backorder_cost = np.array([19.0] + [0.0] * (self.num_stock_points - 1))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.inventory = np.zeros(self.num_stock_points, dtype=np.float32)
        self.backorders = np.zeros(self.num_stock_points, dtype=np.float32)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.t += 1
        action = np.clip(action, 0, self.max_order)
        demand = np.random.poisson(lam=10)
        shipped = min(self.inventory[0], demand)
        backorder = max(demand - shipped, 0)
        self.inventory[0] -= shipped
        self.backorders[0] += backorder
        self.inventory += action
        holding_costs = np.sum(self.holding_cost * np.maximum(self.inventory, 0))
        backorder_costs = np.sum(self.backorder_cost * self.backorders)
        reward = -(holding_costs + backorder_costs)
        obs = self._get_obs()
        done = self.t >= 128
        return obs, reward, done, False, {}

    def _get_obs(self):
        return self.inventory.copy()

    def render(self):
        print(f"t={self.t}, Inventory={self.inventory}, Backorders={self.backorders}")
