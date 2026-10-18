"""Dataset-independent selector evaluation over a complete controller matrix."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from simulation.research.pse import CASE, predict_loso, score_predictions
from simulation.research.pse_consequences import predict_consequences


def json_value(value):
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return 'always_reference' if value==float('inf') else None
    return value


def validate_matrix(features,kpis,policies):
    """Refuse partial cases; seeds and origins cannot silently disappear."""
    from simulation.research.production import SEEDS, STOCHASTIC
    if features.empty:raise ValueError('No feature cases')
    if features.duplicated(CASE).any():raise ValueError('Duplicate feature cases')
    if not np.isfinite(kpis.TIC).all() or kpis.TIC.lt(0).any():raise ValueError('Invalid cost')
    defined=kpis.NS.dropna()
    if not defined.between(0,1).all():raise ValueError('Invalid service')
    if kpis.duplicated(CASE+['policy','seed','block']).any():raise ValueError('Duplicate fit metrics')
    expected={(p,s,b) for p in policies for s in (SEEDS if p in STOCHASTIC else [0]) for b in ['validation','test']}
    wanted={tuple(row) for row in features[CASE].itertuples(index=False,name=None)}
    actual=set()
    for key,g in kpis.groupby(CASE):
        actual.add(key)
        if set(g[['policy','seed','block']].itertuples(index=False,name=None))!=expected:raise ValueError('Incomplete policy/seed/block matrix: '+str(key))
    if actual!=wanted:raise ValueError('Feature and outcome case sets differ')


def evaluate_selectors(features,kpis,output,inner_folds=5):
    """Freeze recommendations before test scoring; no controller is fitted."""
    from simulation.research.production import POLICIES
    policies=list(dict.fromkeys([*POLICIES,'DQN']))
    validate_matrix(features,kpis,policies)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol={'scope':'Development evaluation; independent confirmation requires a separately reserved cohort',
              'inner_folds':inner_folds,'outer':'per-origin leave-one-store-out',
              'alpha':.7,'policies':policies,'cases':len(features),
              'input_features_sha256':hashlib.sha256(features.to_csv(index=False).encode()).hexdigest(),
              'input_kpis_sha256':hashlib.sha256(kpis.to_csv(index=False).encode()).hexdigest()}
    (output/'protocol_before_predictions.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    validation=kpis[kpis.block.eq('validation')]
    controls=predict_loso(features,validation)
    new,folds=predict_consequences(features,validation,inner_folds=inner_folds)
    predictions=pd.concat([controls,new],ignore_index=True)
    if predictions.duplicated(CASE+['strategy']).any():raise ValueError('Duplicate recommendations')
    if not predictions.groupby('strategy').size().eq(len(features)).all():raise ValueError('Missing recommendations')
    predictions.to_csv(output/'predictions_frozen_before_scoring.csv',index=False)
    # Infinity means always reference, serialized explicitly instead of invalid JSON.
    (output/'inner_validation.json').write_text(json.dumps(json_value(folds),indent=2,allow_nan=False),encoding='utf-8')
    scored=score_predictions(predictions,kpis[kpis.block.eq('test')]);scored.to_csv(output/'case_metrics.csv',index=False)
    summary=scored.groupby('strategy').agg(cases=('TIC','size'),mean_cost=('TIC','mean'),mean_service=('NS','mean'),
        defined_service=('NS','count'),feasible=('feasible','sum'),oracle_hits=('oracle_hit','sum'))
    summary.to_csv(output/'summary.csv')
    # One row per independent store cluster/strategy; origins remain paired.
    scored.groupby(['store_id','strategy']).agg(cases=('TIC','size'),mean_cost=('TIC','mean'),mean_service=('NS','mean')).to_csv(output/'store_metrics.csv')
    base=scored[scored.strategy.eq('Global')][CASE+['TIC','NS','policy']]
    paired=scored.merge(base,on=CASE,suffixes=('','_reference'),validate='many_to_one')
    paired['cost_difference']=paired.TIC-paired.TIC_reference
    paired['service_difference']=paired.NS-paired.NS_reference
    paired['intervened']=paired.policy.ne(paired.policy_reference)
    paired['new_service_failure']=paired.NS_reference.ge(.7-1e-9)&paired.NS.lt(.7-1e-9)
    paired.to_csv(output/'paired_reference.csv',index=False)
    intervals=[];rng=np.random.default_rng(20261015)
    for strategy,g in paired.groupby('strategy'):
        for metric in ['cost_difference','service_difference']:
            clusters=g.groupby('store_id')[metric].agg(['sum','count'])
            ix=rng.integers(0,len(clusters),(5000,len(clusters)))
            numerator=clusters['sum'].to_numpy()[ix].sum(axis=1)
            denominator=clusters['count'].to_numpy()[ix].sum(axis=1)
            samples=numerator[denominator>0]/denominator[denominator>0]
            intervals.append({'strategy':strategy,'metric':metric,'mean':float(g[metric].mean()),
                'ci95_low':float(np.quantile(samples,.025)) if len(samples) else None,
                'ci95_high':float(np.quantile(samples,.975)) if len(samples) else None,
                'store_clusters':len(clusters),'conditional':True})
    (output/'paired_intervals.json').write_text(json.dumps(json_value(intervals),indent=2,allow_nan=False),encoding='utf-8')
    return summary


def evaluate_oracles(cases,kpis,output):
    """Resumable perfect-information plans, separate from selection fitting."""
    import sqlite3
    from simulation.research.production import production_config
    from simulation.research.protocol import common_initial_state
    from simulation.research.oracles import perfect_information_plan, efficiency_percent
    output=Path(output)
    with sqlite3.connect(output/'oracle_cache.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS plans (case_hash TEXT PRIMARY KEY, payload TEXT)')
        records=[]
        for meta,key,demand in cases:
            origin=meta['origin'];identity=dict(zip(CASE,[origin,*key]))
            train=demand[:origin];valid=demand[origin:origin+7];test=demand[origin+7:origin+14]
            cfg=production_config(train,0);_,test_cfg=common_initial_state(train,valid,cfg)
            digest=hashlib.sha256(json.dumps([identity,test.tolist(),test_cfg],sort_keys=True).encode()).hexdigest()
            cached=db.execute('SELECT payload FROM plans WHERE case_hash=?',[digest]).fetchone()
            if cached:result=json.loads(cached[0])
            else:
                result=perfect_information_plan(test,test_cfg)
                with db:db.execute('INSERT INTO plans VALUES (?,?)',(digest,json.dumps(result)))
            records.append(identity|{'physical_target_feasible':result['target_feasible'],
                'perfect_cost':result['cost'],'max_service':result['max_service'],'proven_optimal':result['proven_optimal']})
    plans=pd.DataFrame(records);plans.to_csv(output/'perfect_information.csv',index=False)
    scored=pd.read_csv(output/'case_metrics.csv',dtype={k:str for k in CASE[1:]})
    enriched=scored.merge(plans,on=CASE,validate='many_to_one')
    enriched['failure_cause']=np.select([enriched.NS.isna(),~enriched.physical_target_feasible,
        enriched.oracle_service.lt(.7-1e-9),~enriched.feasible],
        ['demanda_zero','limite_fisico','lacuna_portfolio','erro_selecao'],default='atendeu')
    enriched.groupby(['strategy','failure_cause']).size().unstack(fill_value=0).to_csv(output/'failure_decomposition.csv')
    predictions=pd.read_csv(output/'predictions_frozen_before_scoring.csv',dtype={k:str for k in CASE[1:]})
    realized=predictions[CASE+['strategy','policy']].merge(kpis[kpis.block.eq('test')],on=CASE+['policy'],validate='many_to_many').merge(plans,on=CASE,validate='many_to_one')
    feasible=realized.NS.ge(.7-1e-9)&realized.physical_target_feasible&realized.proven_optimal
    realized['theory_comparable']=feasible
    realized['theory_efficiency_percent']=[efficiency_percent(r.perfect_cost,r.TIC) if r.theory_comparable else np.nan for r in realized.itertuples()]
    realized['theory_hit']=feasible&np.isclose(realized.TIC,realized.perfect_cost,rtol=1e-6,atol=1e-8)
    realized['physical_attainment_percent']=100*realized.NS/realized.max_service.replace(0,np.nan)
    realized.to_csv(output/'realized_theory_comparisons.csv',index=False)
