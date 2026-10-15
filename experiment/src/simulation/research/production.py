"""Frozen production budgets; opt-in, independent of historical experiment configs."""
import contextlib
import io
import math
import numpy as np
from simulation.research.protocol import calibration_config, DETERMINISTIC, fit_policy
from simulation.core.inventory_env import InventoryEnv
from simulation.research.portfolio import ACTIVE_POLICIES

POLICIES = list(ACTIVE_POLICIES)
STOCHASTIC = {'GA', 'SA', 'DQN', 'PPO', 'GA-DQN'}
SEEDS = [11, 29, 47]
TRANSITIONS = 20000


def fits_per_case(policies=None, seeds=None):
    """Count fitted controllers, rather than assuming the old 12-arm portfolio."""
    policies = POLICIES if policies is None else policies
    seeds = SEEDS if seeds is None else seeds
    return sum(len(seeds) if name in STOCHASTIC else 1 for name in policies)


def production_config(train, seed):
    cfg = calibration_config(train, seed)
    cfg['GENETIC_ALGORITHM'].update(population_size=100, n_generations=50,
                                    bounds_mode='training_scaled')
    cfg['SA'] = {'n_chains':32, 'maxiter':156, 'seed':seed}
    # Same environment-transition budget across datasets/origins, rounded to whole episodes.
    episodes = math.ceil(TRANSITIONS / len(train))
    for name in ['DQN', 'PPO']:
        cfg[name]['episodes'] = episodes
    cfg['HYBRID'] = {'prepopulate_transitions':1000, 'floor_ratio':.5}
    return cfg


def fit_production(name, train, cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        if name == 'SA':
            from simulation.core.metaheuristics_torch import TorchSA
            opt = TorchSA(train, cfg); opt.optimize(verbose=False)
            return opt.make_policy()
        if name in ['CappedBaseStock', 'BigDataNewsvendor']:
            from simulation.core.policies_sota import CappedBaseStockPolicy, BigDataNewsvendorPolicy
            cls = CappedBaseStockPolicy if name == 'CappedBaseStock' else BigDataNewsvendorPolicy
            return cls(train, cfg)
        if name == 'GA-DQN':
            from simulation.core.policies import GAPolicyOptimizer
            from simulation.research.hybrid import GAGuidedDQN, HYBRID_VERSION
            opt = GAPolicyOptimizer(train, cfg); opt.optimize(verbose=False)
            agent = GAGuidedDQN(InventoryEnv.STATE_DIM, cfg, opt.best)
            cfg['HYBRID'].update(implementation=HYBRID_VERSION,ga_parameters=agent.ga)
            agent.prepopulate(train, cfg,n_transitions=cfg['HYBRID']['prepopulate_transitions'])
            agent.train(train, cfg, verbose=False)
            return agent.make_policy()
        return fit_policy(name, train, cfg)
