"""Canonical BA/M5 plan -> fit -> evaluate, reusing audited checkpoints.

Default operation is a non-training plan. Brazil is deliberately not a target.
Historical runners remain for provenance; this entrypoint owns new executions.
"""
from pathlib import Path
import argparse
import ast
from collections import Counter
import json
import os
import shutil
import sqlite3
import sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import run_production_v2 as queue
from reuse_production_checkpoints import WithoutDocstrings
from simulation.research.production import POLICIES, SEEDS, STOCHASTIC, production_config, fits_per_case
from simulation.research.parallel import ParallelFitPool
from simulation.research.hybrid import HYBRID_VERSION

POLICIES=list(dict.fromkeys([*POLICIES,'DQN']))
BASE=ROOT/'experiment/data/08_reporting/research_v2'
SOURCES={'BA':['bahia145_cpu16_2026-10-15'],
         'M5':['m5_pse_pilot30_v2_2026-10-15','m5_zabraoui2408_cpu16_2026-10-15']}


def semantic(path, exclude_calibrator=False):
    tree=WithoutDocstrings().visit(ast.parse(path.read_text(encoding='utf-8-sig')))
    if exclude_calibrator:
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='_calibrate_batch']
    return ast.dump(tree,include_attributes=False)


def audit_source(source):
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    capped_ok=True;checked=[]
    for name,digest in manifest['code_sha256'].items():
        normalized=name.replace('\\','/')
        relevant='/core/' in normalized or normalized.endswith(('research/protocol.py','research/production.py','research/parallel.py','research/hybrid.py','demand_profiling/nodes.py'))
        if not relevant:continue
        old=source/'code_snapshot'/name;current=ROOT/name
        if queue.sha256(old)!=digest:raise ValueError('Snapshot integrity failure: '+name)
        if semantic(old)!=semantic(current):
            if normalized.endswith('core/policies_sota.py') and semantic(old,True)==semantic(current,True):
                capped_ok=False
            else:raise ValueError('Changed controller dependency: '+name)
        checked.append(name)
    if not checked:raise ValueError('No scientific dependencies audited')
    return {'source':source.name,'capped_compatible':capped_ok,'checked':checked}


def expected_config(train,seed,payload,policy):
    cfg=production_config(train,seed)
    for field in ['SIMULATION','DQN','PPO']:
        cfg[field]['device']=payload['config'][field].get('device','auto')
    if policy=='GA-DQN':
        hybrid=payload['config']['HYBRID']
        if hybrid.get('implementation')!=HYBRID_VERSION:return None
        if any(hybrid.get(k)!=v for k,v in cfg['HYBRID'].items()):return None
        cfg['HYBRID']=hybrid
    return cfg


def plan(dataset,output):
    parent=BASE/SOURCES[dataset][-1]
    metadata=json.loads((parent/'panels'/f'{dataset}_{dataset}.json').read_text(encoding='utf-8'))
    panels={}
    for meta in metadata:
        path=parent/meta['panel']
        if queue.sha256(path)!=meta['panel_sha256']:raise ValueError('Frozen panel changed')
        panels[meta['origin']]=pd.read_parquet(path)
    reusable={};audits=[];rejected=Counter();times={p:[] for p in POLICIES}
    for name in SOURCES[dataset]:
        source=BASE/name;audit=audit_source(source);audits.append(audit)
        pilot_cases={}
        if (source/'cases.json').exists():
            for c in json.loads((source/'cases.json').read_text(encoding='utf-8')):
                pilot_cases[(c['origin'],*(str(c[k]) for k in queue.KEYS))]=c
        with sqlite3.connect((source/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            for row in db.execute('SELECT * FROM fits'):
                tid,ds,uf,origin,policy,seed,seconds,raw=row
                if ds!=dataset or policy not in POLICIES:continue
                times[policy].append(seconds)
                if tid in reusable:continue
                if policy=='CappedBaseStock' and not audit['capped_compatible']:
                    rejected['obsolete_capped_calibration']+=1;continue
                payload=json.loads(raw);rows=payload['rows']
                key=tuple(str(rows[0][k]) for k in queue.KEYS)
                if origin not in panels or key not in panels[origin].index:raise ValueError('Out-of-cohort checkpoint')
                demand=panels[origin].loc[key].to_numpy(float)
                if pilot_cases:
                    case=pilot_cases[(origin,*key)]
                    if not np.array_equal(np.concatenate([case['train'],case['validation'],case['test']]),demand[:origin+14]):raise ValueError('Pilot data differs')
                if tid!=queue.task_id(ds,uf,origin,key,policy,seed):raise ValueError('Checkpoint identity differs')
                if len(rows)!=2 or {r['block'] for r in rows}!={'validation','test'}:raise ValueError('Incomplete checkpoint')
                if payload['config']!=expected_config(demand[:origin],seed,payload,policy):
                    rejected['configuration_mismatch']+=1;continue
                reusable[tid]={'source':name,'payload_sha256':__import__('hashlib').sha256(raw.encode()).hexdigest(),'policy':policy}
    total_cases=sum(len(p) for p in panels.values());used=Counter(r['policy'] for r in reusable.values())
    pending={p:total_cases*(len(SEEDS) if p in STOCHASTIC else 1)-used[p] for p in POLICIES}
    work=sum(pending[p]*(float(np.median(times[p])) if times[p] else 0) for p in POLICIES)
    signature={str(p.relative_to(ROOT)):queue.sha256(p) for p in [Path(__file__),Path(queue.__file__),Path(__file__).with_name('reuse_production_checkpoints.py'),*sorted((ROOT/'experiment/src/simulation').rglob('*.py'))]}
    result={'dataset':dataset,'state':'planned_not_started','policies':POLICIES,'seeds':SEEDS,
        'cohorts':[{'origin':o,'series':len(p),'stores':p.index.get_level_values('store_id').nunique()} for o,p in panels.items()],
        'series_origin_cases':total_cases,'planned_fits':total_cases*fits_per_case(POLICIES),'reusable_fits':len(reusable),
        'reuse_by_policy':dict(used),'pending_by_policy':pending,'remaining_fits':sum(pending.values()),
        'execution':{'workers':8,'device':'cpu','priority':'below-normal','threads_per_worker':1},
        'linear_fit_hours_at_8_workers':work/8/3600,'estimate_note':'Historical median fit duration; excludes selectors, startup and load variation; not an ETA guarantee.',
        'audits':audits,'rejected':dict(rejected),'code_sha256':signature,'panels_source':parent.name,
        'panel_metadata':metadata,'reusable_tasks':reusable,
        'evaluation':{'outer':'leave-one-store-out, separate origin','inner_group_folds':5,
                      'pilot':'development; overlapping series must not be claimed independent confirmation'}}
    output.mkdir(parents=True,exist_ok=True)
    queue.write_json(output/'plan.json',result)
    return result


def execute(output,planned):
    if planned['state']!='planned_not_started':raise ValueError('Invalid plan')
    for name,digest in planned['code_sha256'].items():
        if queue.sha256(ROOT/name)!=digest:raise ValueError('Code changed after plan; plan again')
    if (output/'cancelled_by_user.json').exists():raise ValueError('Cancelled run')
    import msvcrt
    global_lock=(BASE/'aipe_active_queue.lock').open('a+b');global_lock.write(b'0');global_lock.flush();global_lock.seek(0)
    msvcrt.locking(global_lock.fileno(),msvcrt.LK_NBLCK,1)
    lock=(output/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    dataset=planned['dataset'];parent=BASE/planned['panels_source']
    (output/'panels').mkdir(exist_ok=True)
    for meta in planned['panel_metadata']:
        original=parent/meta['panel'];dest=output/meta['panel'];dest.parent.mkdir(parents=True,exist_ok=True)
        if queue.sha256(original)!=meta['panel_sha256']:raise ValueError('Panel changed after planning')
        shutil.copy2(original,dest)
    queue.write_json(output/'panels'/f'{dataset}_{dataset}.json',planned['panel_metadata'])
    for name in planned['code_sha256']:
        dest=output/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    db=queue.open_results(output/'results.sqlite')
    queue.write_json(output/'manifest.json',{k:v for k,v in planned.items() if k!='reusable_tasks'})
    try:
        for source in SOURCES[dataset]:
            with sqlite3.connect((BASE/source/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as old:
                for row in old.execute('SELECT * FROM fits'):
                    expected=planned['reusable_tasks'].get(row[0])
                    if not expected or expected['source']!=source:continue
                    if __import__('hashlib').sha256(row[-1].encode()).hexdigest()!=expected['payload_sha256']:raise ValueError('Source payload changed')
                    payload=json.loads(row[-1]);payload['canonical_reuse']={'source':source,'source_payload_sha256':expected['payload_sha256']}
                    with db:db.execute('INSERT OR IGNORE INTO fits VALUES (?,?,?,?,?,?,?,?)',(*row[:-1],json.dumps(payload,allow_nan=False)))
        status={'state':'running','dataset':dataset,'planned_fits':planned['planned_fits'],
                'completed_fits':db.execute('SELECT count(*) FROM fits').fetchone()[0],'execution':planned['execution'],'pid':os.getpid()}
        tagged=((dataset,meta,key,values) for meta,key,values in queue.cases(planned['panel_metadata'],output))
        if status['completed_fits']<planned['planned_fits']:
            with ParallelFitPool(8,'cpu',priority='below-normal') as pool:
                queue.run_parallel(tagged,pool,db,output,status,policies=POLICIES)
        queue.export_completed(db,output)
        status['state']='controllers_completed';queue.write_json(output/'status.json',status)
    except BaseException as exc:
        queue.write_json(output/'status.json',{'state':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            'dataset':dataset,'completed_fits':db.execute('SELECT count(*) FROM fits').fetchone()[0],
            'planned_fits':planned['planned_fits'],'error':repr(exc),'execution':planned['execution']})
        raise
    finally:db.close();lock.close()
    try:evaluate(output,planned)
    except BaseException as exc:
        queue.write_json(output/'status.json',{'state':'evaluation_failed','dataset':dataset,
            'completed_fits':planned['planned_fits'],'planned_fits':planned['planned_fits'],'error':repr(exc)})
        raise
    finally:global_lock.close()


def evaluate(output,planned):
    from simulation.research.evaluation import evaluate_selectors
    records=[]
    for meta,key,values in queue.cases(planned['panel_metadata'],BASE/planned['panels_source']):
        from simulation.research.protocol import profile_of
        from simulation.pipelines.demand_profiling.nodes import _series_features
        records.append(dict(zip(['origin',*queue.KEYS],[meta['origin'],*key]))|_series_features(values[:meta['origin']],17)|{'operational_profile':profile_of(values[:meta['origin']])})
    evaluate_selectors(pd.DataFrame(records),pd.read_csv(output/'kpis.csv',dtype={k:str for k in queue.KEYS}),output/'selection')
    if planned['dataset']=='M5':
        # Report the development cohort separately; this flag is not itself proof of independence.
        pilot=pd.read_csv(BASE/'m5_pse_pilot30_v2_2026-10-15/features.csv',dtype={k:str for k in queue.KEYS})
        metrics=pd.read_csv(output/'selection/case_metrics.csv',dtype={k:str for k in queue.KEYS})
        membership=pd.MultiIndex.from_frame(pilot[queue.KEYS].drop_duplicates())
        metrics['development_pilot_series']=pd.MultiIndex.from_frame(metrics[queue.KEYS]).isin(membership)
        metrics.groupby(['development_pilot_series','strategy']).agg(cases=('TIC','size'),mean_cost=('TIC','mean'),mean_service=('NS','mean'),feasible=('feasible','sum')).to_csv(output/'selection/pilot_vs_remaining_cohort.csv')
    from simulation.research.visualizations import generate_research_visualizations
    generate_research_visualizations(output,datasets=(planned['dataset'],))
    from simulation.research.evaluation import evaluate_oracles
    cases=queue.cases(planned['panel_metadata'],BASE/planned['panels_source'])
    evaluate_oracles(cases,pd.read_csv(output/'kpis.csv',dtype={k:str for k in queue.KEYS}),output/'selection')
    queue.write_json(output/'status.json',{'state':'completed','dataset':planned['dataset'],
        'completed_fits':planned['planned_fits'],'planned_fits':planned['planned_fits'],'execution':planned['execution']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['plan','execute','evaluate'],nargs='?',default='plan')
    parser.add_argument('--dataset',choices=['BA','M5'],required=True);parser.add_argument('--run',required=True);args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    output=BASE/args.run
    if args.phase=='plan':
        if (output/'results.sqlite').exists():raise ValueError('Do not replace plan of an execution already started')
        result=plan(args.dataset,output)
        print(json.dumps({k:v for k,v in result.items() if k not in ['code_sha256','reusable_tasks','panel_metadata','audits']},indent=2))
    else:
        planned=json.loads((output/'plan.json').read_text(encoding='utf-8'))
        if planned['dataset']!=args.dataset:raise ValueError('Dataset differs from frozen plan')
        for name,digest in planned['code_sha256'].items():
            if queue.sha256(ROOT/name)!=digest:raise ValueError('Code changed after plan; use a fresh plan before execution')
        (execute if args.phase=='execute' else evaluate)(output,planned)


if __name__=='__main__':main()
