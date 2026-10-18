"""Independent BA145, historical M5 membership, and national experiment queues."""
from pathlib import Path
import argparse
import ast
from copy import deepcopy
import json
import os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
import shutil
import sqlite3
import traceback
import pandas as pd

import run_production_v2 as queue
from reuse_production_checkpoints import semantic,WithoutDocstrings
from simulation.research.production import POLICIES, SEEDS, production_config, fits_per_case
from simulation.research.hybrid import HYBRID_VERSION
from simulation.research.parallel import ParallelFitPool
from simulation.research.visualizations import generate_research_visualizations

ROOT=queue.ROOT
PARENT=ROOT/'experiment/data/08_reporting/research_v2'


def policy_body_without_hybrid(path):
    tree=WithoutDocstrings().visit(ast.parse(path.read_text(encoding='utf-8')))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_production')
    class StripHybrid(ast.NodeTransformer):
        def visit_If(self,node):
            if ast.unparse(node.test)=="name == 'GA-DQN'":return None
            return self.generic_visit(node)
    return ast.dump(StripHybrid().visit(function),include_attributes=False)


def audit_source(source):
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if status['state'] not in ['stopped_scope_correction','stopped_by_user','completed','failed']:raise ValueError('Source is still active: '+source.name)
    snapshot=source/'code_snapshot'
    for name in manifest['code_sha256']:
        normalized=name.replace('\\','/')
        if '/core/' in normalized or normalized.endswith(('research/protocol.py','demand_profiling/nodes.py','data_ingestion/m5_loader.py')):
            if semantic(snapshot/name)!=semantic(ROOT/name):raise ValueError('Scientific dependency changed: '+name)
    production='experiment/src/simulation/research/production.py'
    if semantic(snapshot/production,{'production_config'})!=semantic(ROOT/production,{'production_config'}):raise ValueError('Training budgets changed')
    if policy_body_without_hybrid(snapshot/production)!=policy_body_without_hybrid(ROOT/production):raise ValueError('Nonhybrid fitting changed')
    runner='experiment/scripts/analysis/run_production_v2.py'
    names={'read_real','prepare','cases','task_id'}
    if semantic(snapshot/runner,names)!=semantic(ROOT/runner,names):raise ValueError('Data/evaluation protocol changed')
    hybrid=snapshot/'experiment/src/simulation/research/hybrid.py'
    same_hybrid=hybrid.exists() and semantic(hybrid)==semantic(ROOT/'experiment/src/simulation/research/hybrid.py')
    return manifest,same_hybrid


def prepare_panels(dataset,source,output):
    target=output/'panels';target.mkdir(exist_ok=True)
    if dataset=='BR':
        for path in (source/'panels').glob('BR_*'):
            shutil.copy2(path,target/path.name)
        return {'membership':'All past-eligible national series; no series cap','members':None}
    sub='' if dataset=='BA' else 'm5/'
    membership_path=ROOT/f'experiment/data/04_feature/{sub}demand_profiles.parquet'
    membership=pd.read_parquet(membership_path)[queue.KEYS].astype(str)
    expected_count=145 if dataset=='BA' else 2408
    if len(membership)!=expected_count or membership.duplicated().any():raise ValueError('Unexpected historical membership')
    index=pd.MultiIndex.from_frame(membership)
    source_meta=json.loads((source/'panels'/f'{dataset}_{dataset}.json').read_text(encoding='utf-8'))
    last=max(source_meta,key=lambda m:m['origin'])
    source_panel=source/last['panel']
    if queue.sha256(source_panel)!=last['panel_sha256']:raise ValueError('Source panel changed')
    full=pd.read_parquet(source_panel)
    if not index.isin(full.index).all():raise ValueError('Historical cohort missing from corrected raw panel')
    full=full.loc[index];metadata=[]
    membership.to_csv(output/'cohort_membership.csv',index=False)
    for old in source_meta:
        origin=old['origin'];positive=(full.iloc[:,:origin]>0).sum(axis=1)
        eligible=full.loc[positive>=3].sort_index()
        path=target/f'{dataset}_{dataset}_{origin}.parquet';eligible.to_parquet(path)
        meta=deepcopy(old)
        meta.update(parent_candidate_series=old['candidate_series'],candidate_series=len(full),
                    eligible_series=len(eligible),evaluated_series_planned=len(eligible),
                    zero_positive_prefix=int((positive==0).sum()),one_or_two_positive_prefix=int(((positive>0)&(positive<3)).sum()),
                    panel=str(path.relative_to(output)),panel_sha256=queue.sha256(path),
                    cohort_membership_sha256=queue.sha256(output/'cohort_membership.csv'),
                    cohort_scope='Fixed historical IDs; corrected raw demand; eligibility only from the training prefix')
        metadata.append(meta)
    queue.write_json(target/f'{dataset}_{dataset}.json',metadata)
    return {'membership_source':str(membership_path.relative_to(ROOT)),'membership_source_sha256':queue.sha256(membership_path),
            'members':len(membership),'cohorts':[{'origin':m['origin'],'eligible_series':m['eligible_series']} for m in metadata],
            'note':'Retrospectively selected membership; not fully prospective cohort selection. M5 CV>=1.5 is a local adaptation, not an exactly reproduced Zabraoui sample.'}


def import_checkpoints(dataset,sources,output,policies):
    db=queue.open_results(output/'results.sqlite');panels={};imported={};invalid_hybrid=0;out_of_scope=0
    for source in sources:
        manifest,same_hybrid=audit_source(source)
        source_db=sqlite3.connect(f'{(source/"results.sqlite").as_uri()}?mode=ro',uri=True)
        source_dataset='BA' if dataset=='BR' else dataset
        for tid,_,uf,origin,policy,seed,seconds,text in source_db.execute('SELECT * FROM fits WHERE dataset=?',[source_dataset]):
            if policy not in policies:continue
            if policy=='GA-DQN' and not same_hybrid:invalid_hybrid+=1;continue
            payload=json.loads(text);rows=payload['rows'];key=tuple(str(rows[0][k]) for k in queue.KEYS)
            target_id=queue.task_id(dataset,uf,origin,key,policy,seed)
            if db.execute('SELECT 1 FROM fits WHERE task_id=?',[target_id]).fetchone():continue
            panel_key=(uf,origin)
            if panel_key not in panels:
                path=output/'panels'/f'{dataset}_{uf}_{origin}.parquet'
                if not path.exists():out_of_scope+=1;continue
                panels[panel_key]=pd.read_parquet(path)
            panel=panels[panel_key]
            if key not in panel.index:out_of_scope+=1;continue
            train=panel.loc[key].to_numpy(float)[:origin]
            expected=production_config(train,seed)
            for field in ['SIMULATION','DQN','PPO']:
                preference=payload['config'][field].get('device','auto')
                if preference not in ['auto','cpu','cuda']:raise ValueError('Unknown source device')
                expected[field]['device']=preference
            if policy=='GA-DQN':
                if payload['config']['HYBRID'].get('implementation')!=HYBRID_VERSION:raise ValueError('Hybrid version mismatch')
                for field,value in expected['HYBRID'].items():
                    if payload['config']['HYBRID'].get(field)!=value:raise ValueError('Hybrid budget changed')
                expected['HYBRID']=payload['config']['HYBRID']
            if payload['config']!=expected:raise ValueError('Checkpoint configuration mismatch: '+tid)
            if len(rows)!=2 or {r['block'] for r in rows}!={'validation','test'}:raise ValueError('Incomplete source checkpoint')
            previous=payload.get('checkpoint_provenance')
            for row in rows:row['dataset']=dataset;row['reused_from']=source.name
            payload['checkpoint_provenance']={'source_run':source.name,'source_task_id':tid,
                'source_payload_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),
                'source_device_inventory':manifest.get('device',manifest.get('execution')),
                'upstream':previous,'note':'Original outputs preserved; no CPU/CUDA numerical identity claim.'}
            with db:db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?,?)',
                [target_id,dataset,uf,origin,policy,seed,seconds,json.dumps(payload,allow_nan=False)])
            imported[policy]=imported.get(policy,0)+1
        source_db.close()
    count=db.execute('SELECT count(*) FROM fits').fetchone()[0];db.close()
    report={'sources':[s.name for s in sources],'imported_by_policy':imported,'completed_fits':count,
            'old_hybrid_checkpoints_not_reused':invalid_hybrid,'out_of_scope_not_imported':out_of_scope,
            'audit':'Unchanged shared core, protocol, data functions, nonhybrid fit branches and exact per-fit configuration. GA-DQN reuse requires new variant identity.'}
    queue.write_json(output/'reused_checkpoints.json',report);return report


def update_execution_profile(manifest,signature,execution,output):
    """Explicit operational migration; scientific source hashes stay unchanged."""
    runner_key=str(Path(__file__).resolve().relative_to(ROOT))
    previous=manifest['code_sha256']
    changed={name for name in set(previous)|set(signature) if previous.get(name)!=signature.get(name)}
    if changed-{runner_key}:raise ValueError('Scientific code changed; cannot migrate execution profile')
    if runner_key in changed:
        old=output/'code_snapshot'/runner_key
        if queue.sha256(old)!=previous[runner_key]:raise ValueError('Previous runner snapshot changed')
        protected={'policy_body_without_hybrid','audit_source','prepare_panels','import_checkpoints'}
        if semantic(old,protected)!=semantic(Path(__file__),protected):raise ValueError('Data/checkpoint semantics changed')
        archive=output/'operational_runner_history'/previous[runner_key]
        archive.mkdir(parents=True,exist_ok=True);shutil.copy2(old,archive/'run_scoped_production.py')
        shutil.copy2(Path(__file__),old)
    manifest.setdefault('execution_history',[]).append({'changed_utc':queue.utcnow(),
        'previous_execution':manifest['execution'],'new_execution':execution,
        'previous_runner_sha256':previous.get(runner_key),'new_runner_sha256':signature.get(runner_key),
        'reason':'Explicit user request: fewer CPU processes and reduced priority; completed checkpoints retained.'})
    manifest['execution']=execution;manifest['code_sha256']=signature
    queue.write_json(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--dataset',choices=['BA','M5','BR'],required=True)
    parser.add_argument('--source',action='append',required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--priority',choices=['normal','below-normal'],default='normal')
    parser.add_argument('--resume-execution-profile',action='store_true')
    parser.add_argument('--include-dqn',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--reuse-completed-bahia',help='Additional completed Bahia queue to reuse in the national stage')
    args=parser.parse_args()
    os.environ['AIPE_DEVICE']='cpu'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.priority=='below-normal':
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for name in [args.run]+args.source+([args.reuse_completed_bahia] if args.reuse_completed_bahia else []):
        if not name.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run name')
    if not 1<=args.workers<=61:raise ValueError('Invalid worker count')
    output=PARENT/args.run
    if (output/'cancelled_by_user.json').exists():raise SystemExit('Run explicitly cancelled by user; refusing to start')
    output.mkdir(exist_ok=True)
    import msvcrt
    lock=(output/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    policies=list(POLICIES)+(['DQN'] if args.include_dqn else [])
    source_files=[Path(__file__),Path(queue.__file__)]+sorted((ROOT/'experiment/src/simulation').rglob('*.py'))
    signature={str(p.relative_to(ROOT)):queue.sha256(p) for p in source_files}
    execution={'workers':args.workers,'device':'cpu','priority':args.priority,'threads_per_worker':1,'max_pending':2*args.workers}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['policies']!=policies or manifest['dataset']!=args.dataset:raise ValueError('Frozen policies or dataset differ')
        if manifest['code_sha256']!=signature or manifest['execution']!=execution:
            if not args.resume_execution_profile:raise ValueError('Frozen run differs; use explicit operational migration only if scientific code is unchanged')
            update_execution_profile(manifest,signature,execution,output)
    else:
        sources=[PARENT/name for name in args.source]
        scope=prepare_panels(args.dataset,sources[0],output)
        reuse=import_checkpoints(args.dataset,sources,output,policies)
        manifest={'created_utc':queue.utcnow(),'status':'queued','dataset':args.dataset,'execution':execution,
            'policies':policies,'primary_policies':list(POLICIES),'control_policies':['DQN'] if args.include_dqn else [],
            'independent_training_seeds':SEEDS,'fits_per_series_origin':fits_per_case(policies),
            'cohort_scope':scope,'reused_checkpoints':reuse,'hybrid_version':HYBRID_VERSION,
            'code_sha256':signature,'cohorts':[],
            'limitations':['Fixed historical BA/M5 IDs are a retrospective cohort. Corrected demand and past-only fit do not make membership prospective.',
                'New hybrid has the same fixed GA gate/floor in demonstrations, training and inference, and a consistent terminal reward; old hybrid results are excluded.',
                'Original compatible checkpoints retain backend provenance. Repeated origins and seeds are not independent retail units.']}
        queue.write_json(manifest_path,manifest)
        for path in source_files:
            target=output/'code_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    db=queue.open_results(output/'results.sqlite')
    if args.dataset=='BR' and args.reuse_completed_bahia and not args.prepare_only:
        additional=PARENT/args.reuse_completed_bahia
        extra=import_checkpoints('BR',[additional],output,policies)
        manifest['additional_completed_bahia_reuse']=extra
        queue.write_json(manifest_path,manifest)
    status={'state':'queued' if args.prepare_only else 'preparing','pid':os.getpid(),'started_utc':queue.utcnow(),
            'dataset':args.dataset,'execution':execution,'completed_fits':db.execute('SELECT count(*) FROM fits').fetchone()[0]}
    cached=[]
    for path in sorted((output/'panels').glob(f'{args.dataset}_*.json')):cached.extend(json.loads(path.read_text(encoding='utf-8')))
    status['planned_fits']=sum(m['eligible_series'] for m in cached)*fits_per_case(policies)
    queue.write_json(output/'status.json',status)
    if args.prepare_only:
        print(json.dumps(status),flush=True);db.close();lock.close();return
    import ctypes
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));status['sleep_prevention_requested']=awake
    try:
        with ParallelFitPool(args.workers,'cpu',priority=args.priority) as pool:
            ufs=[args.dataset] if args.dataset!='BR' else [uf for uf in queue.UFS if (ROOT/'data/source/vendas'/('uf='+uf)).is_dir()]
            for uf in ufs:
                metadata=queue.prepare(args.dataset,uf,output)
                manifest['cohorts']=[m for m in manifest['cohorts'] if m['uf']!=uf]+metadata
                manifest['status']='running';queue.write_json(manifest_path,manifest)
                print('Cohorts',args.dataset,uf,[(m['origin'],m['eligible_series']) for m in metadata],flush=True)
                tagged=((args.dataset,*case) for case in queue.cases(metadata,output))
                queue.run_parallel(tagged,pool,db,output,status,policies)
        queue.export_completed(db,output)
        if args.dataset!='BR':generate_research_visualizations(output,datasets=(args.dataset,))
        status.update(state='completed',finished_utc=queue.utcnow());queue.write_json(output/'status.json',status)
        manifest.update(status='completed',finished_utc=queue.utcnow());queue.write_json(manifest_path,manifest)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),updated_utc=queue.utcnow());queue.write_json(output/'status.json',status)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);db.close();lock.close()


if __name__=='__main__':main()
