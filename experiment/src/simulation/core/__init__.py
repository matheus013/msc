"""Exportacoes do nucleo carregadas sob demanda, sem importar toda a pilha ML."""
from importlib import import_module

_EXPORTS = {'InventoryEnv': 'simulation.core.inventory_env', 'EOQPolicy': 'simulation.core.policies', 'SsPolicyClass': 'simulation.core.policies', 'NewsvendorPolicy': 'simulation.core.policies', 'GAPolicyOptimizer': 'simulation.core.policies', 'SimulatedAnnealingPolicy': 'simulation.core.policies', 'PSOPolicy': 'simulation.core.policies', 'DEPolicy': 'simulation.core.policies', 'DQNPolicy': 'simulation.core.policies', 'PPOPolicy': 'simulation.core.policies', 'SARSAPolicy': 'simulation.core.policies', 'HybridGADQN': 'simulation.core.policies', 'HybridGAPPO': 'simulation.core.policies', 'LSTMNumpy': 'simulation.core.forecasting', 'ANNForecaster': 'simulation.core.forecasting', 'XGBoostForecaster': 'simulation.core.forecasting', 'train_all_forecasters': 'simulation.core.forecasting', 'evaluate_forecaster': 'simulation.core.forecasting'}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
