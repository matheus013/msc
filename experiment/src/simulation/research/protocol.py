"""Calibration/validation/test helpers shared by the research-v2 runner."""
from copy import deepcopy
import contextlib
import io
import numpy as np
from simulation.core.inventory_env import InventoryEnv
from simulation.core.policies import EOQPolicy,SsPolicyClass,NewsvendorPolicy,GAPolicyOptimizer
from simulation.core.policies_zabraoui import FixedIntervalPolicy,VendorResponsivePolicy
from simulation.pipelines.demand_profiling.nodes import _series_features,classify_operational_profiles
import pandas as pd

DETERMINISTIC={'EOQCycle':EOQPolicy,'sSCycle':SsPolicyClass,'NewsvendorHeuristic':NewsvendorPolicy,
               'FixedInterval':FixedIntervalPolicy,'VendorResponsive':VendorResponsivePolicy}


def calibration_config(train,seed=42):
    d=np.asarray(train,float);mu=float(d.mean());sd=float(d.std())
    return {'SIMULATION':{'lead_time':2,'lead_time_convention':'exact','initial_inventory':mu*2+1.28*sd*np.sqrt(2),
            'causal_state':True,'reference_mean':mu,'reference_std':sd,'history_demand':d[-6:].tolist(),
            'device':'auto','alpha_min':.7},
            'COST':{'holding_cost_per_unit':1.,'stockout_cost_per_unit':5.,'ordering_cost_per_order':50.,'ordering_cost_per_unit':.5},
            'HEURISTIC':{'z_score':1.28,'eoq_time_basis':'per_cycle'},
            'GENETIC_ALGORITHM':{'population_size':32,'n_generations':8,'seed':seed,'fitness_mode':'constrained',
              'alpha_min':.7,'penalty_weight':30,'tr_weight':0.,'be_weight':0.},
            'DQN':{'episodes':10,'seed':seed,'device':'auto','n_actions':20,'max_order_qty':max(1.,mu*12),
                   'batch_size':32,'hidden_layers':[64,64],'memory_size':10000},
            'PPO':{'episodes':10,'seed':seed,'device':'auto','n_actions':20,'max_order_qty':max(1.,mu*12),
                   'hidden_layers':[64,64],'update_epochs':4}}


def fit_policy(name,train,cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        if name in DETERMINISTIC:return DETERMINISTIC[name](train,cfg)
        if name=='GA':
            optimizer=GAPolicyOptimizer(train,cfg);optimizer.optimize(verbose=False);return optimizer.make_policy()
        if name in ['DQN','PPO']:
            from simulation.core.policies import DQNPolicy,PPOPolicy
            agent=(DQNPolicy if name=='DQN' else PPOPolicy)(InventoryEnv.STATE_DIM,cfg)
            agent.train(train,cfg,verbose=False);return agent.make_policy()
    raise ValueError(name)


def simulate(demand,cfg,policy):
    env=InventoryEnv(np.asarray(demand,float),cfg);state=env.reset();done=False
    while not done:state,_,done,_=env.step(policy(state,env))
    k=env.kpis()
    k['ServiceLevel']=env._demand_served/env._demand_total if env._demand_total else float('nan')
    k['DemandTotal']=env._demand_total;k['DemandServed']=env._demand_served
    k['TerminalInventory']=env.inventory;k['TerminalPipeline']=sum(env.pipeline)
    k['TerminalCommittedUnits']=env.inventory+sum(env.pipeline)
    return k,env


def common_initial_state(train,validation,cfg):
    """Known-past baseline history gives all candidate tests the same state.

    Candidate validation episodes start from the same neutral warm-up state.
    Test states come from a fixed EOQ baseline, not candidate-specific validation
    inventory. Neither state uses test demand or the eventual selected policy.
    """
    reference=fit_policy('EOQCycle',train,cfg)
    warm=deepcopy(cfg);warm['SIMULATION']['history_demand']=list(np.asarray(train[:-6],float)[-6:])
    _,env=simulate(np.asarray(train,float)[-6:],warm,reference)
    valid_cfg=deepcopy(cfg)
    valid_cfg['SIMULATION'].update(initial_inventory=env.inventory,initial_pipeline=env.pipeline.copy(),history_demand=list(np.asarray(train,float)[-6:]))
    _,env=simulate(validation,valid_cfg,reference)
    test_cfg=deepcopy(cfg)
    test_cfg['SIMULATION'].update(initial_inventory=env.inventory,initial_pipeline=env.pipeline.copy(),
                                 history_demand=list(np.concatenate([train,validation])[-6:]))
    return valid_cfg,test_cfg


def profile_of(train):
    features=_series_features(np.asarray(train,float),17)
    return classify_operational_profiles(pd.DataFrame([features]),{}).operational_profile.iloc[0]
