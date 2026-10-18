import numpy as np
import pandas as pd
import pytest
from simulation.research.pse import CASE
from simulation.research.evaluation import validate_matrix
from simulation.research.pse_consequences import tune_margin


def test_canonical_command_works_outside_project_directory(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'experiment/run.py'),'research-aipe','--help'],
                          cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert '{plan,execute,evaluate}' in result.stdout


def test_partial_matrix_is_rejected_instead_of_silent_case_exclusion():
    identity=dict(zip(CASE,[17,'BA','shop','item']))
    features=pd.DataFrame([identity])
    kpis=pd.DataFrame([identity|{'policy':'GA','seed':11,'block':b,'TIC':1.,'NS':.8} for b in ['validation','test']])
    with pytest.raises(ValueError,match='Incomplete'):validate_matrix(features,kpis,['GA'])
    kpis=pd.concat([kpis,kpis.assign(seed=29),kpis.assign(seed=47)],ignore_index=True)
    validate_matrix(features,kpis,['GA'])


def test_undefined_validation_service_cannot_justify_intervention():
    identity=dict(zip(CASE,[17,'BA','shop','item']))
    predicted=pd.DataFrame([identity|{'policy':'p','TIC':10.,'NS':.9}])
    truth=predicted.assign(NS=np.nan)
    margin,_=tune_margin([(predicted,'p')],truth)
    assert margin==float('inf')


def test_grouped_inner_validation_never_fits_on_target_stores(monkeypatch):
    import simulation.research.pse_consequences as module
    features=pd.DataFrame([dict(zip(CASE,[17,'BA',f's{i}',f'item{j}'])) for i in range(6) for j in range(1+i%2)])
    validation=features.assign(policy='p',TIC=10.,NS=.8)
    fitted=[]
    def fake_fit(train,held,outcomes,with_profile=True):
        assert set(train.store_id).isdisjoint(held.store_id)
        assert set(outcomes.store_id)<=set(train.store_id)
        fitted.append((set(train.store_id),set(held.store_id)))
        return held[CASE].assign(policy='p',TIC=10.,NS=.8)
    monkeypatch.setattr(module,'outcome_predictions',fake_fit)
    monkeypatch.setattr(module,'matched_classifier',lambda train,held,truth:held[CASE].assign(policy='p',strategy='Classe_floresta_controle'))
    predictions,folds=module.predict_consequences(features,validation,inner_folds=3)
    assert predictions.groupby('strategy').size().eq(len(features)).all()
    assert len(folds)==6 and all(x['inner_folds']==3 for x in folds)
    assert all(x['inner_training_cases_min']<=x['inner_training_cases_max'] for x in folds)
    assert fitted
