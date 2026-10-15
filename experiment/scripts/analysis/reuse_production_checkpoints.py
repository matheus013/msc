"""Audit and import compatible completed fits into a fresh production run.

Does not retrain or relabel old CPU/CUDA computations as new CPU results.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.production import POLICIES,SEEDS,STOCHASTIC,production_config


class WithoutDocstrings(ast.NodeTransformer):
    def strip(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
        return node
    visit_Module=strip
    visit_FunctionDef=strip
    visit_AsyncFunctionDef=strip
    visit_ClassDef=strip


def semantic(path,functions=None):
    tree=WithoutDocstrings().visit(ast.parse(path.read_text(encoding='utf-8')))
    if functions is not None:tree=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions],type_ignores=[])
    return ast.dump(tree,include_attributes=False)


def reuse(source,destination):
    if (destination/'manifest.json').exists():raise ValueError('Destination must be a fresh run')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if status['state'] not in ['stopped_by_user','failed','completed']:raise ValueError('Source must be stopped')
    snapshot=source/'code_snapshot'
    audited=[]
    for name in manifest['code_sha256']:
        normalized=name.replace('\\','/')
        if '/core/' in normalized or normalized.endswith(('research/protocol.py','demand_profiling/nodes.py','data_ingestion/m5_loader.py')):
            if semantic(snapshot/name)!=semantic(ROOT/name):raise ValueError('Scientific implementation changed: '+name)
            audited.append(name)
    for name,functions in [
        ('experiment/src/simulation/research/production.py',{'production_config','fit_production'}),
        ('experiment/scripts/analysis/run_production_v2.py',{'read_real','prepare','cases','task_id'})]:
        if semantic(snapshot/name,functions)!=semantic(ROOT/name,functions):raise ValueError('Protocol functions changed: '+name)
        audited.append(name)
    source_db=sqlite3.connect(f'{(source/"results.sqlite").as_uri()}?mode=ro',uri=True)
    spec=importlib.util.spec_from_file_location('queue_import',ROOT/'experiment/scripts/analysis/run_production_v2.py')
    queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    imported=[];panels={};excluded=0
    for tid,dataset,uf,origin,policy,seed,seconds,payload_text in source_db.execute('SELECT * FROM fits'):
        if policy not in POLICIES or seed not in (SEEDS if policy in STOCHASTIC else [0]):excluded+=1;continue
        payload=json.loads(payload_text);rows=payload['rows']
        if len(rows)!=2 or {row['block'] for row in rows}!={'validation','test'}:raise ValueError('Incomplete checkpoint '+tid)
        key=tuple(str(rows[0][k]) for k in queue.KEYS)
        if tid!=queue.task_id(dataset,uf,origin,key,policy,seed):raise ValueError('Task identity mismatch '+tid)
        panel_key=(dataset,uf,origin)
        if panel_key not in panels:
            path=source/'panels'/f'{dataset}_{uf}_{origin}.parquet'
            meta=json.loads((source/'panels'/f'{dataset}_{uf}.json').read_text(encoding='utf-8'))
            expected=next(m for m in meta if m['origin']==origin)
            if queue.sha256(path)!=expected['panel_sha256']:raise ValueError('Source panel changed')
            panels[panel_key]=pd.read_parquet(path)
        train=panels[panel_key].loc[key].to_numpy(float)[:origin]
        expected_config=production_config(train,seed)
        if payload['config']!=expected_config:raise ValueError('Training configuration changed '+tid)
        for row in rows:
            if tuple(str(row[k]) for k in queue.KEYS)!=key or row['policy']!=policy or row['seed']!=seed:raise ValueError('Payload mismatch')
            row['reused_from']=source.name
        payload['checkpoint_provenance']={'source_run':source.name,'source_task_id':tid,
            'source_payload_sha256':hashlib.sha256(payload_text.encode()).hexdigest(),
            'source_device_preference':payload['config']['SIMULATION'].get('device'),
            'source_device_inventory':manifest.get('device'),
            'note':'Original completed evaluation retained; CPU/CUDA RNG equivalence is not claimed.'}
        imported.append((tid,dataset,uf,origin,policy,seed,seconds,json.dumps(payload,allow_nan=False)))
    source_count=source_db.execute('SELECT count(*) FROM fits').fetchone()[0];source_db.close()
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'panels',destination/'panels',dirs_exist_ok=True)
    if (source/'national_plan.json').exists():shutil.copy2(source/'national_plan.json',destination/'national_plan.json')
    db=queue.open_results(destination/'results.sqlite')
    with db:db.executemany('INSERT OR IGNORE INTO fits VALUES (?,?,?,?,?,?,?,?)',imported)
    db.close()
    audit={'source_run':source.name,'source_manifest_sha256':queue.sha256(source/'manifest.json'),
           'source_completed_fits':source_count,'imported_fits':len(imported),'excluded_retired_policy_fits':excluded,
           'audited_scientific_sources':audited,'configuration_check':'Exact per-checkpoint production_config equality against the preserved training panel.',
           'backend_note':'Earlier auto-device fits retain original outputs and device provenance. All newly scheduled fits use the new run device; no cross-device numerical identity is assumed.'}
    queue.write_json(destination/'reused_checkpoints.json',audit)
    return audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    for name in [args.source,args.destination]:
        if not name.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run name')
    parent=ROOT/'experiment/data/08_reporting/research_v2'
    print(json.dumps(reuse(parent/args.source,parent/args.destination),indent=2))


if __name__=='__main__':main()
