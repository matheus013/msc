"""Resumable small M5 pilot; eight CPU workers, audited fit reuse, LOSO PSE."""
from pathlib import Path
import argparse
import ast
import json
import os
import shutil
import sqlite3
import sys
import time
import traceback
for variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
import numpy as np
import pandas as pd
import run_production_v2 as queue
from simulation.research.parallel import ParallelFitPool
from simulation.research.production import POLICIES,SEEDS,STOCHASTIC,production_config
from simulation.research.protocol import common_initial_state,profile_of
from simulation.pipelines.demand_profiling.nodes import _series_features
from simulation.research.pse import CASE,predict_loso,score_predictions,best_policy,paired_store_intervals
from simulation.research.oracles import perfect_information_plan,efficiency_percent

# DQN is retained as the matched control for the GA-DQN hybrid.
POLICIES=list(dict.fromkeys([*POLICIES,'DQN']))

SOURCE=ROOT/'experiment/data/08_reporting/research_v2/m5_zabraoui2408_cpu16_2026-10-15'


def semantic(path,skip_calibration=False):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    if skip_calibration:tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='_calibrate_batch']
    return ast.dump(tree,include_attributes=False)


def audit_reuse():
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    audited=[]
    for name in manifest['code_sha256']:
        normalized=name.replace('\\','/')
        if '/src/simulation/' not in normalized:continue
        if queue.sha256(SOURCE/'code_snapshot'/name)!=manifest['code_sha256'][name]:
            raise ValueError('Source snapshot integrity failure: '+name)
        if semantic(SOURCE/'code_snapshot'/name,normalized.endswith('policies_sota.py'))!=semantic(ROOT/name,normalized.endswith('policies_sota.py')):
            raise ValueError('Changed dependency: '+name)
        audited.append(name)
    if not audited:raise ValueError('No source code audited')
    return {'source':SOURCE.name,'audited_files':audited,'excluded_policy':'CappedBaseStock',
        'exception':'Only _calibrate_batch changed; never used by other six active policies.'}


def sample_keys(panel):
    """Three volume strata per store, only first 50 training cycles."""
    rng=np.random.default_rng(20261015);keys=[]
    for store,group in panel.groupby(level='store_id',sort=True):
        ranked=group.iloc[:,:50].mean(axis=1).sort_values(kind='stable')
        for positions in np.array_split(np.arange(len(ranked)),3):
            keys.append(tuple(ranked.index[int(rng.choice(positions))]))
    return keys


def prepare(output):
    audit=audit_reuse()
    metadata=json.loads((SOURCE/'panels/M5_M5.json').read_text(encoding='utf-8'))
    panels={}
    for meta in metadata:
        path=SOURCE/meta['panel']
        if queue.sha256(path)!=meta['panel_sha256']:raise ValueError('Changed frozen panel')
        panels[meta['origin']]=pd.read_parquet(path)
    keys=sample_keys(panels[50]);jobs=[];features=[];cases=[]
    for origin,panel in sorted(panels.items()):
        for key in keys:
            demand=panel.loc[key].to_numpy(float)
            train=demand[:origin];validation=demand[origin:origin+7];test=demand[origin+7:origin+14]
            case=dict(zip(CASE,[origin,*key]));cases.append(case|{'train':train.tolist(),'validation':validation.tolist(),'test':test.tolist()})
            features.append(case|_series_features(train,17)|{'operational_profile':profile_of(train)})
            for policy in POLICIES:
                for seed in SEEDS if policy in STOCHASTIC else [0]:
                    task=queue.task_id('M5','M5',origin,key,policy,seed)
                    jobs.append({'job_id':task,'dataset':'M5','uf':'M5','origin':origin,'key':key,
                        'policy':policy,'seed':seed,'train':train.tolist(),'validation':validation.tolist(),'test':test.tolist()})
    files=[Path(__file__),ROOT/'experiment/scripts/analysis/run_production_v2.py']+sorted((ROOT/'experiment/src/simulation').rglob('*.py'))
    signature={str(p.relative_to(ROOT)):queue.sha256(p) for p in files}
    manifest={'protocol_version':1,'purpose':'Development pilot, not confirmatory final evidence',
        'sampling':'3 first-prefix volume strata per store; one seeded draw per stratum; fixed historical candidate cohort',
        'sample_seed':20261015,'series':len(keys),'stores':10,'origins':sorted(panels),'cases':len(cases),
        'planned_fits':len(jobs),'policies':POLICIES,'training_seeds':SEEDS,'alpha':.7,
        'execution':{'workers':8,'device':'cpu','priority':'below-normal','threads_per_worker':1},
        'source_panels':metadata,'reuse_audit':audit,'code_sha256':signature,
        'validation':'Per-origin leave-one-store-out; validation-only labels, training-only features, fold-local preprocessing; no rare-class deletion',
        'oracles':'Test-only portfolio hindsight; perfect-information deterministic MILP compared per realized seed only when service feasible',
        'limitations':['Small sample conditional on eligibility at first origin; no cold-start coverage',
            'Ten store clusters only; repeated origins and seeds are dependent',
            'Historical candidate membership is not an exact Zabraoui replication']}
    path=output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=manifest:raise ValueError('Pilot changed: choose a new run name')
    else:
        queue.write_json(path,manifest)
        for file in files:
            target=output/'code_snapshot'/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    queue.write_json(output/'cases.json',cases)
    pd.DataFrame(features).to_csv(output/'features.csv',index=False)
    return jobs,cases,pd.DataFrame(features),manifest


def evaluate(db,output,cases,features):
    queue.export_completed(db,output)
    kpis=pd.read_csv(output/'kpis.csv',dtype={'warehouse':str,'store_id':str,'item_id':str})
    predictions=predict_loso(features,kpis[kpis.block.eq('validation')])
    predictions.to_csv(output/'predictions_frozen_before_test.csv',index=False)
    test=kpis[kpis.block.eq('test')]
    scored=score_predictions(predictions,test);scored.to_csv(output/'pse_case_metrics.csv',index=False)
    queue.write_json(output/'paired_store_intervals.json',paired_store_intervals(scored))
    # Persist predictions before looking at test demand or hindsight outcomes.
    theoretical=[];realized=[]
    chosen=predictions[predictions.strategy.eq('AIPE_PSE')]
    for case in cases:
        identity={k:case[k] for k in CASE}
        cfg=production_config(case['train'],0)
        _,test_cfg=common_initial_state(case['train'],case['validation'],cfg)
        plan=perfect_information_plan(case['test'],test_cfg)
        theoretical.append(identity|plan)
        mask=np.logical_and.reduce([test[k].eq(v) for k,v in identity.items()])
        candidates=test[mask]
        selected=chosen[np.logical_and.reduce([chosen[k].eq(v) for k,v in identity.items()])].iloc[0].policy
        for seed in SEEDS:
            matched=candidates[(candidates.seed.eq(seed)&candidates.policy.isin(STOCHASTIC))|
                               (candidates.seed.eq(0)&~candidates.policy.isin(STOCHASTIC))]
            row=matched[matched.policy.eq(selected)].iloc[0]
            comparable=plan['proven_optimal'] and plan['target_feasible'] and pd.notna(row.NS) and row.NS>=.7-1e-9
            realized.append(identity|{'training_seed':seed,'policy':selected,'TIC':float(row.TIC),
                'NS':None if pd.isna(row.NS) else float(row.NS),'theory_comparable':bool(comparable),
                'theory_efficiency_percent':efficiency_percent(plan['cost'],row.TIC) if comparable else None,
                'theory_hit':bool(comparable and np.isclose(row.TIC,plan['cost'],rtol=1e-6,atol=1e-8)),
                'physical_fill_attainment_percent':100*row.NS/plan['max_service'] if pd.notna(row.NS) and plan['max_service'] else None})
        queue.write_json(output/'perfect_information.json',theoretical)
    queue.write_json(output/'pse_realized_seed_metrics.json',realized)
    summary={}
    for strategy,group in scored.groupby('strategy'):
        summary[strategy]={'cases':len(group),'mean_cost':float(group.TIC.mean()),'mean_service':float(group.NS.mean()),
            'undefined_service':int(group.NS.isna().sum()),'feasible_cases':int(group.feasible.sum()),
            'oracle_comparable_cases':int(group.comparable.sum()),'oracle_hits':int(group.oracle_hit.sum()),
            'oracle_hit_percent_of_comparable':100*float(group.oracle_hit.sum())/group.comparable.sum() if group.comparable.any() else None,
            'within_1_percent_cases':int(group.within_1_percent.sum()),
            'within_5_percent_cases':int(group.within_5_percent.sum()),
            'mean_oracle_efficiency_percent':float(group.efficiency_percent.mean()) if group.efficiency_percent.notna().any() else None}
    hindsight=[]
    for origin,g in test.groupby('origin'):
        means=g.groupby(CASE+['policy'],as_index=False)[['TIC','NS']].mean()
        hindsight.append({'origin':int(origin),'policy':best_policy(means),'diagnostic_only':True})
    queue.write_json(output/'summary.json',{'strategies':summary,'best_single_test_policy':hindsight,
        'theoretical_optima_proven':sum(p['proven_optimal'] for p in theoretical),
        'theory_comparable_seed_realizations':sum(r['theory_comparable'] for r in realized),
        'theory_hits_seed_realizations':sum(r['theory_hit'] for r in realized),
        'mean_theory_efficiency_percent':float(np.mean([r['theory_efficiency_percent'] for r in realized if r['theory_efficiency_percent'] is not None])) if any(r['theory_comparable'] for r in realized) else None,
        'independent_series':len({tuple(c[k] for k in CASE[1:]) for c in cases}),
        'note':'Seeds and origins are dependent; efficiencies conditional on common service feasibility. Development pilot.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='m5_pse_pilot30_v2_2026-10-15');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    output=ROOT/'experiment/data/08_reporting/research_v2'/args.run;output.mkdir(parents=True,exist_ok=True)
    import msvcrt,psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    lock=(output/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    jobs,cases,features,manifest=prepare(output)
    db=queue.open_results(output/'results.sqlite')
    old=sqlite3.connect(f'file:{(SOURCE/"results.sqlite").as_posix()}?mode=ro',uri=True)
    reused=0
    for job in jobs:
        if job['policy']=='CappedBaseStock' or db.execute('SELECT 1 FROM fits WHERE task_id=?',[job['job_id']]).fetchone():continue
        row=old.execute('SELECT * FROM fits WHERE task_id=?',[job['job_id']]).fetchone()
        if row:
            payload=json.loads(row[-1]);payload['provenance']={'source':SOURCE.name,'source_task_id':row[0]}
            for r in payload['rows']:r['reused_from']=SOURCE.name
            with db:db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?,?)',(*row[:-1],json.dumps(payload,allow_nan=False)))
            reused+=1
    old.close()
    completed={r[0] for r in db.execute('SELECT task_id FROM fits')};pending=[j for j in jobs if j['job_id'] not in completed]
    status={'state':'prepared','planned_fits':len(jobs),'completed_fits':len(completed),'newly_imported_fits':reused,
        'remaining_fits':len(pending),'updated_utc':queue.utcnow(),'pid':os.getpid()}
    queue.write_json(output/'status.json',status);print(json.dumps(status),flush=True)
    if args.prepare_only:return
    try:
        if pending:
            status['state']='running';queue.write_json(output/'status.json',status)
            with ParallelFitPool(8,device='cpu',priority='below-normal') as pool:
                for job,result in pool.results(pending):
                    queue.commit_fit(db,job,result);status['completed_fits']+=1;status['remaining_fits']-=1
                    status.update(policy=job['policy'],origin=job['origin'],updated_utc=queue.utcnow())
                    queue.write_json(output/'status.json',status)
                    print(json.dumps(status),flush=True)
        status['state']='evaluating_pse';queue.write_json(output/'status.json',status)
        evaluate(db,output,cases,features)
        status.update(state='completed',updated_utc=queue.utcnow());queue.write_json(output/'status.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),updated_utc=queue.utcnow());queue.write_json(output/'status.json',status);raise
    finally:db.close()


if __name__=='__main__':main()
