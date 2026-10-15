"""Selection leakage, complete-case comparisons and pipeline wiring."""
import json
import numpy as np
import pandas as pd
from simulation.research.visualizations import (
    pareto_mask,prefix_features,balanced_results,held_store_selections,read_completed_cohort,
)


def fixture_rows():
    rows=[]
    for store in ['a','b']:
        for policy in ['EOQCycle','GA']:
            for seed in ([11,29] if policy=='GA' else [0]):
                for block in ['validation','test']:
                    rows.append({'dataset':'BA','origin':3,'warehouse':'BA','store_id':store,
                                 'item_id':'x','profile':'sparse','policy':policy,'seed':seed,
                                 'block':block,'TIC':1. if policy=='EOQCycle' else 10.,'NS':.8})
    return pd.DataFrame(rows)


def test_pareto_prefers_lower_cost_and_higher_service_but_preserves_ties():
    assert pareto_mask([1,2,3,1,0,np.nan],[.5,.4,.9,.5,np.nan,.7]).tolist()==[True,False,True,True,False,False]


def test_prefix_features_exclude_future_and_keep_zero_demand_undefined():
    wide=pd.DataFrame([[0,2,4,999],[0,0,0,999]])
    features=prefix_features(wide,3)
    assert features.iloc[0].adi==1.5
    assert np.isclose(features.iloc[0].cv2_positive,2/9)
    assert features.iloc[1].zero_fraction==1
    assert np.isnan(features.iloc[1].adi) and np.isnan(features.iloc[1].cv2_positive)


def test_incomplete_or_wrong_seed_sets_cannot_bias_policy_means():
    rows=fixture_rows();rows.loc[(rows.store_id=='b')&(rows.seed==29),'seed']=47
    complete,excluded=balanced_results(rows,['EOQCycle','GA'],[11,29])
    assert excluded==1 and set(complete.store_id)=={'a'}


def test_selector_does_not_use_held_store_validation_or_any_test_costs():
    rows=fixture_rows()
    # Held store a favors GA in validation, but every other store favors EOQ.
    rows.loc[(rows.store_id=='a')&(rows.policy=='GA')&(rows.block=='validation'),'TIC']=0.
    complete,_=balanced_results(rows,['EOQCycle','GA'],[11,29])
    choices=held_store_selections(complete,[.7])
    assert set(choices[choices.store_id=='a'].policy)=={'EOQCycle'}
    rows.loc[(rows.block=='test')&(rows.policy=='GA'),'TIC']=-1000000.
    changed,_=balanced_results(rows,['EOQCycle','GA'],[11,29])
    assert choices.policy.tolist()==held_store_selections(changed,[.7]).policy.tolist()


def test_sqlite_seed_average_agrees_with_dataframe_and_skips_partial(tmp_path):
    import sqlite3
    rows=fixture_rows();rows=rows.drop(rows[(rows.store_id=='b')&(rows.seed==29)&(rows.block=='test')].index)
    db=sqlite3.connect(tmp_path/'fits.sqlite')
    db.execute('CREATE TABLE fits(dataset TEXT, origin INTEGER, policy TEXT, seed INTEGER, payload TEXT)')
    for (store,policy,seed),g in rows.groupby(['store_id','policy','seed']):
        db.execute('INSERT INTO fits VALUES (?,?,?,?,?)',['BA',3,policy,int(seed),json.dumps({'rows':g.to_dict('records')})])
    expected,excluded=balanced_results(rows,['EOQCycle','GA'],[11,29])
    actual,sql_excluded=read_completed_cohort(db,'BA',3,['EOQCycle','GA'],[11,29])
    assert sql_excluded==excluded==1
    assert len(actual)==len(expected)==4
    assert np.allclose(sorted(actual.TIC),sorted(expected.TIC))
    db.close()


def test_five_diagnostics_are_in_all_experiment_and_reporting_pipelines():
    from simulation.pipeline_registry import register_pipelines
    pipelines=register_pipelines()
    for name in ['experiment_diagnostics','reporting','profile_analysis','final_report',
                 'benchmark_final','benchmark_m5','benchmark_bot','__default__']:
        nodes=[node for node in pipelines[name].nodes if node.name=='generate_experiment_diagnostics']
        assert len(nodes)==1,name
        assert 'scenarios' in nodes[0].inputs and 'params:simulation' in nodes[0].inputs
