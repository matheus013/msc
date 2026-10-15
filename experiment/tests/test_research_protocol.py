"""Scientific regressions: causal states, shipping timing and raw sales."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch
from simulation.core.inventory_env import InventoryEnv
from simulation.core.inventory_env_torch import BatchInventoryEnv
from simulation.pipelines.data_ingestion.nodes import clean_sales_data


def cfg():
    return {'SIMULATION': {'lead_time': 2, 'lead_time_convention': 'exact',
            'initial_inventory': 0, 'causal_state': True, 'reference_mean': 2,
            'reference_std': 1, 'history_demand': [1, 2, 3]},
            'COST': {'holding_cost_per_unit':1, 'stockout_cost_per_unit':5,
                     'ordering_cost_per_order':3, 'ordering_cost_per_unit':.5}}


def test_future_suffix_cannot_change_current_state_or_past_rewards():
    a=InventoryEnv(np.array([1,2,3,4]),cfg())
    b=InventoryEnv(np.array([1,2,9000,8000]),cfg())
    assert np.array_equal(a.reset(),b.reset())
    for order in [10,0]:
        sa,ra,_,_=a.step(order);sb,rb,_,_=b.step(order)
        assert ra==rb and np.array_equal(sa,sb)
    with pytest.raises(ValueError,match='reference'):
        broken=cfg();del broken['SIMULATION']['reference_mean'];InventoryEnv(np.array([1,2]),broken)


@pytest.mark.parametrize('lead_time',[0,1,2,3])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_known_order_arrives_exactly_at_configured_cycle(lead_time,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    config=cfg();config['SIMULATION']['lead_time']=lead_time
    demand=np.ones(6)
    scalar=InventoryEnv(demand,config)
    batch=BatchInventoryEnv(demand,config,device=torch.device(device))
    for t in range(6):
        order=10 if t==0 else 0
        scalar.step(order);batch.step(torch.tensor([order],device=device))
        assert scalar.inventory==pytest.approx(batch.inventory[0].item())
        assert scalar._demand_served==pytest.approx(batch.served[0].item())
        if t<lead_time:assert scalar._demand_served==0
        if t==lead_time:assert scalar._demand_served==1


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_causal_scalar_batch_states_match_with_initial_pipeline(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    config=cfg();config['SIMULATION'].update(initial_inventory=6,initial_pipeline=[3,5])
    demand=np.array([2.,4.,1.,7.])
    scalar=InventoryEnv(demand,config);batch=BatchInventoryEnv(demand,config,device=torch.device(device))
    assert np.allclose(scalar.reset(),batch.reset()[0].cpu().numpy())
    for order in [0,8,0,3]:
        state,_,_,_=scalar.step(order);bs,_,done,_=batch.step(torch.tensor([order],device=device))
        if not done:assert np.allclose(state,bs[0].cpu().numpy(),rtol=1e-5,atol=1e-5)
    assert scalar.kpis()['TIC']==pytest.approx(batch.kpis()['TIC'][0].item())


def test_same_cycle_sales_are_kept_and_transaction_conflicts_are_rejected():
    frame=pd.DataFrame({'warehouse':['BA']*2,'store_id':['x']*2,'item_id':['p']*2,
                       'venda_ciclo':[202301]*2,'demand':[2.,7.],'date':['a','b']})
    params={'dedup_strategy':'none','remove_zero_only_stores':False}
    assert clean_sales_data(frame,params).demand.sum()==9
    assert clean_sales_data(frame,{'dedup_strategy':'exact_row'}).demand.sum()==9
    assert clean_sales_data(pd.concat([frame,frame]),{'dedup_strategy':'exact_row'}).demand.sum()==9
    conflicting=frame.assign(transaction='same')
    with pytest.raises(ValueError,match='Conflicting'):
        clean_sales_data(conflicting,{'dedup_strategy':'transaction_id','transaction_id_columns':['transaction']})


def test_eoq_lot_does_not_depend_on_number_of_identical_cycles():
    from simulation.core.policies import EOQPolicy,SsPolicyClass
    config=cfg();config['HEURISTIC']={'z_score':1.28,'eoq_time_basis':'per_cycle'}
    a=np.array([2.,4.,2.,4.]);b=np.tile(a,4)
    ea,eb=EOQPolicy(a,config),EOQPolicy(b,config)
    assert ea.Q==pytest.approx(eb.Q)
    sa,sb=SsPolicyClass(a,config),SsPolicyClass(b,config)
    assert sa.S==pytest.approx(sb.S)


def test_common_test_inventory_is_determined_only_by_known_history():
    from simulation.research.protocol import calibration_config,common_initial_state
    train=np.array([1.,2.,3.,1.,6.,2.,4.,3.]);validation=np.array([2.,1.,8.])
    config=calibration_config(train)
    v,t=common_initial_state(train,validation,config)
    a=InventoryEnv(np.array([1.,2.,3.]),t)
    b=InventoryEnv(np.array([900.,800.,700.]),t)
    assert np.array_equal(a.reset(),b.reset())
    assert a.pipeline==b.pipeline and a.inventory==b.inventory
    assert v['SIMULATION']['reference_mean']==t['SIMULATION']['reference_mean']==train.mean()
