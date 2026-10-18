import numpy as np
import pandas as pd
from simulation.research.pse import CASE
from simulation.research.pse_consequences import decisions, tune_margin, outcome_predictions
from simulation.pipelines.policy_selection.nodes import FEATURE_COLS


def matrix(cost=80, service=.8):
    identity=dict(zip(CASE,[50,'CA','CA_1','item']))
    return pd.DataFrame([identity|{'policy':'reference','TIC':100.,'NS':.75},
                         identity|{'policy':'candidate','TIC':cost,'NS':service}])


def test_guard_requires_both_service_and_cost_and_can_abstain():
    assert decisions(matrix(),'reference',.1,True).iloc[0].policy=='candidate'
    assert decisions(matrix(service=.72),'reference',.1,True).iloc[0].policy=='reference'
    assert decisions(matrix(cost=95),'reference',.1,True).iloc[0].policy=='reference'
    assert decisions(matrix(),'reference',float('inf'),True).iloc[0].policy=='reference'


def test_inner_tuning_rejects_observed_service_deterioration():
    predicted=matrix()
    truth=matrix(cost=50,service=.6)
    margin,_=tune_margin([(predicted,'reference')],truth)
    assert margin==float('inf')


def test_held_validation_outcomes_do_not_affect_model_fit():
    rng=np.random.default_rng(3)
    features=pd.DataFrame(rng.uniform(0,1,(10,len(FEATURE_COLS))),columns=FEATURE_COLS)
    features['origin']=50;features['warehouse']='CA';features['store_id']=[f'CA_{i}' for i in range(10)]
    features['item_id']='item';features['operational_profile']='profile'
    train,held=features.iloc[:9],features.iloc[9:]
    outcomes=[]
    for i,row in features.iterrows():
        for policy in ['reference','candidate']:
            outcomes.append({k:row[k] for k in CASE}|{'policy':policy,'TIC':100+i,'NS':.8})
    truth=pd.DataFrame(outcomes)
    first=outcome_predictions(train,held,truth)
    truth.loc[truth.store_id.eq('CA_9'),['TIC','NS']]=[999999.,0.]
    second=outcome_predictions(train,held,truth)
    pd.testing.assert_frame_equal(first,second)
