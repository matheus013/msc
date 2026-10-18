import numpy as np
import pytest
from simulation.research.protocol import calibration_config
from simulation.research.oracles import perfect_information_plan


def config(initial=0,lead=2):
    cfg=calibration_config(np.array([1.,0.,2.]))
    cfg['SIMULATION'].update(initial_inventory=initial,initial_pipeline=[0.]*lead,lead_time=lead)
    return cfg


def test_perfect_information_plan_matches_known_just_in_time_solution():
    result=perfect_information_plan([0.,0.,3.],config(),alpha=1)
    assert result['proven_optimal']
    assert result['cost']==pytest.approx(51.5)
    assert result['orders']==pytest.approx([3,0,0])
    assert result['service']==pytest.approx(1.)


def test_initial_lead_time_can_make_perfect_service_impossible():
    result=perfect_information_plan([4.,0.,3.],config(),alpha=.7)
    assert not result['target_feasible']
    assert result['max_service']==pytest.approx(3/7)
    assert result['service']==pytest.approx(3/7)
    assert result['cost']==pytest.approx(71.5)


def test_zero_demand_is_not_assigned_perfect_service():
    result=perfect_information_plan([0.,0.,0.],config(initial=2))
    assert result['proven_optimal'] and result['service'] is None
    assert result['max_service'] is None
    assert result['cost']==pytest.approx(6.)


def test_initial_pipeline_is_available_at_correct_cycle():
    cfg=config();cfg['SIMULATION']['initial_pipeline']=[2.,1.]
    result=perfect_information_plan([2.,1.,0.],cfg,alpha=1)
    assert result['cost']==pytest.approx(0)
    assert result['service']==pytest.approx(1.)


def test_zero_lead_time_allows_same_cycle_ordering():
    result=perfect_information_plan([3.],config(lead=0),alpha=1)
    assert result['cost']==pytest.approx(51.5)


def test_fractional_pipeline_does_not_create_phantom_fixed_order_cost():
    # Bahia: the solver returned q=2.86e-7 with y=7.88e-9 for a second
    # order. Executing that residual incorrectly incurred another fixed 50.
    cfg=config();cfg['SIMULATION']['initial_pipeline']=[12.126781251816649,0.]
    result=perfect_information_plan([3.,102.,12.,0.,0.,9.,0.],cfg)
    assert result['proven_optimal']
    assert result['cost']==pytest.approx(560.992875,abs=1e-5)
    assert sum(q>0 for q in result['orders'])==1
    assert result['service']==pytest.approx(result['max_service'],abs=1e-7)


def test_grid_calibration_respects_explicit_risk_weights(monkeypatch):
    import torch
    import simulation.core.policies_sota as sota
    calls=[]
    original=sota.constrained_cost
    def capture(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(sota,'constrained_cost',capture)
    cfg=config();cfg['SIMULATION']['device']='cpu'
    sota._calibrate_batch(np.array([1.,0.,2.]),cfg,
        lambda env,theta:torch.zeros_like(env.inventory),torch.tensor([[0.],[1.]]),device='cpu')
    assert calls and calls[0]['tr_weight']==calls[0]['be_weight']==0.
