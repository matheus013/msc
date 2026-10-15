import importlib.util
from pathlib import Path
import sys
import pandas as pd
from simulation.research.pse import CASE


def diagnostics_module():
    script=Path(__file__).resolve().parents[1]/'scripts/analysis/diagnose_pse_pilot.py'
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('pse_diagnostics',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_guard_uses_training_support_and_score_without_test_inputs():
    module=diagnostics_module();base=[];fold=[]
    for item,score,support in [('1',.59,10),('2',.9,2),('3',.6,3)]:
        identity=dict(zip(CASE,[50,'CA','CA_1',item]))
        for strategy,policy in [('AIPE_PSE','DQN'),('Perfil','sSCycle')]:
            base.append(identity|{'strategy':strategy,'policy':policy})
        fold.append(identity|{'max_class_score':score,'class_support':support})
    result=module.guarded_predictions(pd.DataFrame(base),pd.DataFrame(fold)).set_index('item_id')
    assert result.policy.to_dict()=={'1':'sSCycle','2':'sSCycle','3':'DQN'}
    assert result.used_fallback.to_dict()=={'1':True,'2':True,'3':False}


def test_failure_decomposition_separates_physical_portfolio_and_selector():
    module=diagnostics_module();rows=[];plans=[]
    for item,ns,oracle,physical in [('1',None,None,True),('2',.2,.5,False),
                                    ('3',.4,.5,True),('4',.4,.9,True),('5',.9,.9,True)]:
        key=dict(zip(CASE,[50,'CA','CA_1',item]))
        rows.append(key|{'strategy':'PSE','NS':ns,'oracle_service':oracle,'feasible':ns is not None and ns>=.7})
        plans.append(key|{'target_feasible':physical})
    result=module.service_failure_decomposition(pd.DataFrame(rows),plans)
    assert result.loc['PSE'].to_dict()=={'atendeu':1,'demanda_zero':1,'erro_selecao':1,'lacuna_portfolio':1,'limite_fisico':1}


def test_label_stability_allows_equivalent_ties_without_false_label_switch():
    module=diagnostics_module();key=dict(zip(CASE,[50,'CA','CA_1','1']));rows=[]
    for block,policy,cost in [('validation','A',2.),('validation','B',1.),('test','A',1.),('test','B',1.)]:
        rows.append(key|{'block':block,'policy':policy,'seed':0,'TIC':cost,'NS':.8,'Demand':1.})
    result=module.labels_and_stability(pd.DataFrame(rows)).iloc[0]
    assert not result.same_label
    assert result.same_objective_allowing_ties
