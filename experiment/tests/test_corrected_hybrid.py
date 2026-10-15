import importlib.util
from pathlib import Path
import numpy as np
from simulation.core.inventory_env import InventoryEnv
from simulation.research.production import production_config
from simulation.research.hybrid import GAGuidedDQN
import simulation.research.hybrid as hybrid_module


def agent_and_cfg():
    values=np.array([4.,0.,3.])
    cfg=production_config(values,11)
    for name in ['SIMULATION','DQN','PPO']:cfg[name]['device']='cpu'
    cfg['DQN'].update(episodes=2,batch_size=2)
    return GAGuidedDQN(InventoryEnv.STATE_DIM,cfg,{'ROP':100.,'Q':7.,'SS':0.}),cfg,values


def test_gate_and_floor_are_identical_in_training_and_inference(monkeypatch):
    agent,cfg,values=agent_and_cfg()
    monkeypatch.setattr(agent,'act',lambda state,explore=True:(0,float(agent.actions[0])))
    actual=[]
    class RecordingEnv(InventoryEnv):
        def step(self,quantity):
            actual.append(quantity);return super().step(quantity)
    monkeypatch.setattr(hybrid_module,'InventoryEnv',RecordingEnv)
    agent.episode(values,cfg,0)
    assert actual and set(actual)=={3.5}
    env=InventoryEnv(values,cfg);state=env.reset()
    assert agent.make_policy()(state,env)==agent.executed_quantity(0,env)==3.5
    env.inventory=101.
    assert agent.make_policy()(state,env)==agent.executed_quantity(0,env)==0.


def test_demonstration_action_label_matches_executed_mapping_and_terminal_reward(monkeypatch):
    agent,cfg,values=agent_and_cfg();executed=[]
    class RecordingEnv(InventoryEnv):
        def step(self,quantity):
            result=super().step(quantity);executed.append((quantity,result[1]));return result
    monkeypatch.setattr(hybrid_module,'InventoryEnv',RecordingEnv)
    monkeypatch.setattr(hybrid_module,'_terminal_deficit_penalty',lambda *args:123.)
    transitions=agent.episode(values,cfg,7,demonstration=True)
    for transition,(quantity,reward) in zip(transitions,executed):
        assert quantity==max(agent.actions[transition[1]],3.5)
        assert np.isclose(transition[2],reward-(123. if transition[4] else 0.))
    # Q=7 is not a grid action. The actual demonstration must use its indexed
    # mapped grid action, instead of storing the index but executing continuous Q.
    assert any(not np.isclose(q,7.) for q,_ in executed)


def test_demo_count_and_real_learning_remain_finite():
    agent,cfg,values=agent_and_cfg()
    assert agent.prepopulate(values,cfg,7)==7
    assert len(agent.mem)==7
    agent.train(values,cfg)
    assert len(agent.reward_hist)==2 and np.isfinite(agent.reward_hist).all()
    assert agent.loss_hist and np.isfinite(agent.loss_hist).all()


def test_historical_nonhybrid_fit_contract_is_unchanged():
    script=Path(__file__).resolve().parents[1]/'scripts/analysis/run_scoped_production.py'
    import sys
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('scoped_queue',script)
    queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    source=script.parents[2]/'data/08_reporting/research_v2/overnight_cpu16_2026-10-14'
    _,same_hybrid=queue.audit_source(source)
    assert not same_hybrid
