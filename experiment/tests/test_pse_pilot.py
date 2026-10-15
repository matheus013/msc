import numpy as np
import pandas as pd
import pytest
from simulation.research.pse import CASE,best_policy,predict_loso,score_predictions
from simulation.pipelines.policy_selection.nodes import FEATURE_COLS


def fixture():
    features=[];rows=[]
    for store in range(3):
        for item in range(2):
            key=dict(zip(CASE,[50,'CA',str(store),str(item)]))
            features.append(key|{f:float(item+store) for f in FEATURE_COLS}|{'operational_profile':'rare' if store==2 else 'common'})
            for policy,cost in [('A',10+item),('B',20-item)]:
                rows.append(key|{'policy':policy,'TIC':cost if store<2 else 40-cost,'NS':.8})
    return pd.DataFrame(features),pd.DataFrame(rows)


def test_labels_preserve_zero_demand_and_choose_best_infeasible_service():
    data=pd.DataFrame({'policy':['A','B'],'TIC':[10.,20.],'NS':[np.nan,np.nan]})
    assert best_policy(data)=='A'
    data.NS=[.4,.6]
    assert best_policy(data)=='B'


def test_held_store_validation_cannot_change_its_predictions():
    features,validation=fixture()
    first=predict_loso(features,validation)
    changed=validation.copy();changed.loc[changed.store_id.eq('2'),'TIC']*=1000
    # Held labels can change; predictions must depend only on OTHER stores.
    changed.loc[changed.store_id.eq('2')&changed.policy.eq('A'),'NS']=.1
    second=predict_loso(features,changed)
    cols=CASE+['strategy','policy']
    pd.testing.assert_frame_equal(first[first.store_id.eq('2')][cols].reset_index(drop=True),second[second.store_id.eq('2')][cols].reset_index(drop=True))
    held=first[first.store_id.eq('2')]
    assert len(held)==10 and held.label_unseen_in_training.all()


def test_oracle_does_not_reward_cheap_infeasible_policy_and_counts_ties():
    key=dict(zip(CASE,[50,'CA','1','1']))
    test=pd.DataFrame([key|{'policy':p,'TIC':c,'NS':s} for p,c,s in [('cheap',1.,.1),('A',10.,.8),('B',10.,.8)]])
    predictions=pd.DataFrame([key|{'strategy':p,'policy':p} for p in ['cheap','B']])
    scored=score_predictions(predictions,test).set_index('strategy')
    assert np.isnan(scored.loc['cheap','efficiency_percent'])
    assert not scored.loc['cheap','oracle_hit']
    assert scored.loc['B','oracle_hit']
    assert scored.loc['B','efficiency_percent']==pytest.approx(100.)


def test_pilot_analysis_end_to_end(tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from simulation.research.protocol import common_initial_state,simulate,fit_policy
    from simulation.research.production import production_config
    script=Path(__file__).resolve().parents[1]/'scripts/analysis/run_pse_pilot.py'
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('pilot_runner',script)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    features,_=fixture();cases=[]
    db=runner.queue.open_results(tmp_path/'results.sqlite')
    for _,row in features.iterrows():
        train=[1.,0.,2.,0.,3.]*10;valid=[1.,0.,3.,0.,2.,1.,1.];test=[2.,1.,0.,3.,1.,0.,2.]
        case={k:row[k] for k in CASE}|{'train':train,'validation':valid,'test':test};cases.append(case)
        cfg=production_config(train,0);vc,tc=common_initial_state(train,valid,cfg)
        for policy in ['EOQCycle','sSCycle']:
            fitted=fit_policy(policy,train,cfg);metrics=[]
            for block,values,config in [('validation',valid,vc),('test',test,tc)]:
                k,_=simulate(values,config,fitted)
                metrics.append({'block':block,'TIC':k['TIC'],'NS':k['ServiceLevel']})
            job={'job_id':str(len(cases))+policy,'dataset':'M5','uf':'M5','origin':50,
                 'key':tuple(row[k] for k in CASE[1:]),'policy':policy,'seed':0}
            result={'job_id':job['job_id'],'rows':metrics,'config':cfg,'profile':'common','fit_seconds':0.,
                    'device':'cpu','worker_pid':0,'threads':1,'evaluation_seconds':0.,'total_seconds':0.}
            runner.queue.commit_fit(db,job,result)
    runner.evaluate(db,tmp_path,cases,features)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['independent_series']==6
    assert summary['strategies']['AIPE_PSE']['cases']==6
    assert summary['theoretical_optima_proven']==6
    assert len(json.loads((tmp_path/'paired_store_intervals.json').read_text()))==8
    assert len(pd.read_csv(tmp_path/'predictions_frozen_before_test.csv'))==30
    db.close()
