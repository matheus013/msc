"""Resumable census queue: Bahia + M5, then every known Brazilian UF.

No series cap. One durable SQLite transaction per independently fitted policy/seed.
Run: python experiment/scripts/analysis/run_production_v2.py --run census_2026-10-14
Resume with the same command; changed code/config/source is rejected.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
for _thread_variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_thread_variable]='1'
import shutil
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
import duckdb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiment/src'))
from simulation.research.production import POLICIES, STOCHASTIC, SEEDS, production_config, fits_per_case
from simulation.research.portfolio import PORTFOLIO_STATUS, HYBRID_ABLATION_POLICIES
from simulation.core.device import available_devices
from simulation.research.parallel import ParallelFitPool
from simulation.pipelines.data_ingestion.m5_loader import load_m5_as_internal

KEYS = ['warehouse', 'store_id', 'item_id']
ORIGINS = {'BA':[17,23], 'M5':[50,64,78], 'BR':[17,23]}
UFS = 'AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO'.split()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacement of an open file.
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt==7:
                if path.name=='status.json':
                    alternate=path.with_name(f'status_latest_{os.getpid()}.json')
                    alternate.write_text(tmp.read_text(encoding='utf-8'),encoding='utf-8')
                    print(f'Status temporarily locked; latest state in {alternate.name}',file=sys.stderr,flush=True)
                    return  # telemetry cannot invalidate a committed scientific fit
                raise
            time.sleep(min(.05*2**attempt,.5))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_real(uf, output):
    """Aggregate raw sales on disk; only then zero-fill past-eligible series."""
    files = sorted((ROOT/'data/source/vendas'/('uf='+uf)).glob('*.parquet'))
    if not files:
        raise FileNotFoundError('No raw partition for '+uf)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=4')
    con.execute('SET temp_directory=?', [str(output/'duckdb_tmp')])
    source = [str(p) for p in files]
    con.execute('''CREATE TEMP TABLE sales AS
        SELECT CAST(revendedor_cod AS VARCHAR) store_id,
               CAST(produto_cod AS VARCHAR) item_id, CAST(venda_ciclo AS VARCHAR) period,
               sum(try_cast(venda_qtd AS DOUBLE)) demand
        FROM read_parquet(?, hive_partitioning=false, union_by_name=true)
        WHERE try_cast(venda_qtd AS DOUBLE)>=0 AND isfinite(try_cast(venda_qtd AS DOUBLE)) AND venda_ciclo IS NOT NULL
          AND revendedor_cod IS NOT NULL AND produto_cod IS NOT NULL
        GROUP BY ALL''', [source])
    observed_cycles = [r[0] for r in con.execute('SELECT DISTINCT period FROM sales ORDER BY period').fetchall()]
    # Common business calendar: some states have no sales at all in early cycles.
    # Their first observed cycle must not become calendar position zero.
    cycles = [f'{year}{cycle:02d}' for year in [2023,2024] for cycle in range(1,18)] + [f'2025{cycle:02d}' for cycle in range(1,5)]
    if set(observed_cycles)-set(cycles):
        raise ValueError(f'{uf}: unexpected business cycles; review calendar before execution')
    count = con.execute('SELECT count(*) FROM (SELECT DISTINCT store_id,item_id FROM sales)').fetchone()[0]
    meta = {'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in files},
            'uf':uf, 'cycles':cycles, 'observed_cycles':observed_cycles,'candidate_series':count,
            'aggregation':'Sum complementary raw rows; no aggregate-key deduplication, CV or full-history activity filter.'}
    cohorts = []
    for origin in ORIGINS['BR']:
        con.execute('''CREATE OR REPLACE TEMP TABLE eligible AS
            SELECT store_id,item_id FROM sales WHERE period<=? AND demand>0
            GROUP BY ALL HAVING count(*)>=3''',[cycles[origin-1]])
        n = con.execute('SELECT count(*) FROM eligible').fetchone()[0]
        npositive = con.execute('''SELECT count(*) FROM (SELECT DISTINCT store_id,item_id FROM sales
                                 WHERE period<=? AND demand>0)''',[cycles[origin-1]]).fetchone()[0]
        long = con.execute('''SELECT ? warehouse,s.store_id,s.item_id,s.period,s.demand
                             FROM sales s JOIN eligible USING(store_id,item_id)''',[uf]).df()
        wide = long.pivot(index=KEYS, columns='period', values='demand').reindex(columns=cycles).fillna(0).sort_index()
        meta_origin = meta | {'origin':origin, 'eligible_series':n, 'evaluated_series_planned':n,
                             'zero_positive_prefix':count-npositive,
                             'one_or_two_positive_prefix':npositive-n, 'sampled':False}
        cohorts.append((origin, wide, meta_origin))
    con.close()
    return cohorts


def prepare(dataset, uf, output):
    cache = output/'panels';cache.mkdir(exist_ok=True)
    label = dataset+'_'+uf
    ready = cache/(label+'.json')
    if ready.exists():
        metadata = json.loads(ready.read_text(encoding='utf-8'))
        inputs=sorted((ROOT/'data/m5_walmart').glob('*.csv')) if dataset=='M5' else sorted((ROOT/'data/source/vendas'/('uf='+uf)).glob('*.parquet'))
        if {str(p.relative_to(ROOT)) for p in inputs} != set(metadata[0]['source_files']):
            raise ValueError('Raw file inventory changed: '+label)
        for m in metadata:
            for name,digest in m['source_files'].items():
                if sha256(ROOT/name) != digest:
                    raise ValueError('Raw source changed: '+name)
        return metadata
    if dataset == 'M5':
        source = ROOT/'data/m5_walmart'
        long = load_m5_as_internal({'cycles_per_year':17,'m5':{'path':str(source),
            'categories':['FOODS'], 'max_cycles':92, 'max_series':None,
            'iqr_outlier_smoothing':False, 'with_revenue':False}})
        for key in KEYS+['venda_ciclo']:long[key]=long[key].astype(str)
        wide = long.pivot(index=KEYS, columns='venda_ciclo', values='demand').sort_index(axis=1).fillna(0).sort_index()
        if wide.shape[1]!=92:raise ValueError('Expected 92 full M5 cycles')
        inputs = {str(p.relative_to(ROOT)):sha256(p) for p in source.glob('*.csv')}
        cohorts=[]
        for origin in ORIGINS['M5']:
            positive=(wide.iloc[:,:origin]>0).sum(axis=1)
            eligible=wide.loc[positive>=3]
            cohorts.append((origin,eligible,{'source_files':inputs,'origin':origin,
                'candidate_series':len(wide),'eligible_series':len(eligible),
                'evaluated_series_planned':len(eligible),'zero_positive_prefix':int((positive==0).sum()),
                'one_or_two_positive_prefix':int(((positive>0)&(positive<3)).sum()),'sampled':False,
                'aggregation':'All FOODS series; complete 21-day cycles, no IQR clipping or future filter.'}))
    else:
        cohorts=read_real(uf,output)
    metadata=[]
    for origin,wide,meta in cohorts:
        path=cache/f'{label}_{origin}.parquet'
        wide.to_parquet(path)
        metadata.append(meta|{'dataset':dataset,'uf':uf,'panel':str(path.relative_to(output)),
                             'panel_sha256':sha256(path),'validation_cycles':7,'test_cycles':7})
    write_json(ready,metadata)
    return metadata


def task_id(dataset, uf, origin, key, policy, seed):
    return hashlib.sha256(json.dumps([dataset,uf,origin,list(key),policy,seed]).encode()).hexdigest()


def open_results(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('''CREATE TABLE IF NOT EXISTS fits
       (task_id TEXT PRIMARY KEY, dataset TEXT, uf TEXT, origin INTEGER,
        policy TEXT, seed INTEGER, fit_seconds REAL, payload TEXT)''')
    db.execute('CREATE INDEX IF NOT EXISTS fits_cohort ON fits(dataset,uf,origin)')
    db.commit()
    return db


def cases(metadata, output):
    for meta in metadata:
        path=output/meta['panel']
        if sha256(path)!=meta['panel_sha256']:raise ValueError('Panel changed: '+str(path))
        wide=pd.read_parquet(path)
        for key,row in wide.iterrows():
            yield meta,tuple(str(k) for k in key),row.to_numpy(float)


def pending_jobs(tagged_cases, db, policies=None):
    """Read cases lazily; committed fits are skipped before process submission."""
    for dataset,meta,key,values in tagged_cases:
        origin=meta['origin'];uf=meta['uf']
        for policy in (POLICIES if policies is None else policies):
            for seed in SEEDS if policy in STOCHASTIC else [0]:
                tid=task_id(dataset,uf,origin,key,policy,seed)
                if db.execute('SELECT 1 FROM fits WHERE task_id=?',[tid]).fetchone():continue
                yield {'job_id':tid,'dataset':dataset,'uf':uf,'origin':origin,'key':key,
                       'policy':policy,'seed':seed,'train':values[:origin].tolist(),
                       'validation':values[origin:origin+7].tolist(),
                       'test':values[origin+7:origin+14].tolist()}


def commit_fit(db, job, result):
    """Only the parent calls this, one durable transaction per successful fit."""
    if result['job_id']!=job['job_id']:raise ValueError('Worker task mismatch')
    rows=[]
    for metrics in result['rows']:
        rows.append(dict(zip(KEYS,job['key']))|{field:job[field] for field in
            ['dataset','uf','origin','policy','seed']}|{'profile':result['profile'],
            'fit_seconds':result['fit_seconds'],'reused_from':None}|metrics)
    payload={'rows':rows,'config':result['config'],'execution':{field:result[field]
        for field in ['device','worker_pid','threads','evaluation_seconds','total_seconds']}}
    with db:
        db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?,?)',
            [job['job_id'],job['dataset'],job['uf'],job['origin'],job['policy'],job['seed'],
             result['fit_seconds'],json.dumps(payload,allow_nan=False)])


def run_parallel(tagged_cases, pool, db, output, status, policies=None):
    status.update(state='running',updated_utc=utcnow());write_json(output/'status.json',status)
    for job,result in pool.results(pending_jobs(tagged_cases,db,policies)):
        commit_fit(db,job,result)
        status['completed_fits']+=1
        status.update({field:job[field] for field in ['dataset','uf','origin','policy','seed']})
        status.update(series=list(job['key']),last_completed_fit_seconds=result['fit_seconds'],updated_utc=utcnow())
        write_json(output/'status.json',status)
        print(f"{utcnow()} {job['dataset']}/{job['uf']}/{job['origin']} {job['policy']} seed={job['seed']} fit={result['fit_seconds']:.2f}s completed={status['completed_fits']}",flush=True)


def export_completed(db, output):
    """Stream metrics to CSV without materializing national results in memory."""
    import csv
    path=output/'kpis.csv'
    with path.with_suffix('.csv.tmp').open('w',newline='',encoding='utf-8') as stream:
        writer=None
        for (payload,) in db.execute('SELECT payload FROM fits ORDER BY dataset,uf,origin,task_id'):
            for row in json.loads(payload)['rows']:
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
    os.replace(path.with_suffix('.csv.tmp'),path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--priority',choices=['normal','below-normal'],default='below-normal')
    parser.add_argument('--preflight',action='store_true',help='One synthetic case, all currently selected policies, full budgets; not research evidence')
    args=parser.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.priority=='below-normal':
        import psutil
        process=psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 10)
    execution={'workers':args.workers,'device':args.device,'priority':args.priority,
               'threads_per_worker':1,'max_pending':2*args.workers}
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run name')
    output=ROOT/'experiment/data/08_reporting/research_v2'/args.run
    output.mkdir(parents=True,exist_ok=True)
    # OS lock is released on process death; a second launcher cannot share the GPU queue.
    import msvcrt
    lock=(output/'queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    source_files=[Path(__file__)]+sorted((ROOT/'experiment/src/simulation').rglob('*.py'))
    signature={str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    manifest_path=output/'manifest.json'
    available_ufs=[uf for uf in UFS if (ROOT/'data/source/vendas'/('uf='+uf)).is_dir()]
    missing_ufs=sorted(set(UFS)-set(available_ufs))
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['code_sha256']!=signature or manifest['preflight']!=args.preflight or manifest.get('execution')!=execution:
            raise ValueError('Code/protocol changed; use a new run name instead of mixing results')
    else:
        manifest={'created_utc':utcnow(),'preflight':args.preflight,'status':'queued',
            'execution':execution,
            'series_scope':'All past-eligible series, no sampling or series cap; BA raw; M5 FOODS; then BR every available identified UF.',
            'available_ufs':available_ufs,'missing_ufs':missing_ufs,
            'eligibility':'At least three positive cycles in training prefix only; cold-start histories counted separately.',
            'device':available_devices(),'policies':POLICIES,'independent_training_seeds':SEEDS,
            'fits_per_series_origin':fits_per_case(),
            'portfolio_status':PORTFOLIO_STATUS,'required_hybrid_ablation':HYBRID_ABLATION_POLICIES,
            'origins':ORIGINS,'example_production_config':production_config(np.ones(17),SEEDS[0]),
            'code_sha256':signature,'cohorts':[],'planned_stages':['BA+M5 interleaved','BR by UF'],
            'limitations':['Costs and lead time are research assumptions, not audited business estimates.',
                'POD retains the legacy all-cycle CV-squared definition as a labelled comparator.',
                'No terminal salvage credit; committed terminal stock exported.',
                'Literature-supported candidate portfolio; DQN matched hybrid ablation remains required before any hybrid gain claim.',
                'No forecast or supervised PSE claim. Raw rows are summed, transaction IDs not independently reconciled.',
                'M5 scope is FOODS, not nonfood categories. Unknown UF partition cannot be assigned to a state.',
                'Raw sales are event data: absent records are zero-filled on the common 202301–202504 calendar; source availability gaps are not independently audited.',
                'A successful fit is checkpointed; a fit interrupted mid-training is restarted from its seed.']}
        reused=output/'reused_checkpoints.json'
        if reused.exists():
            manifest['reused_checkpoints']=json.loads(reused.read_text(encoding='utf-8'))
            manifest['limitations'].append('Earlier compatible fits retain their original backend and outputs; new fits use the declared execution device. CPU/CUDA RNG equivalence is not claimed.')
        write_json(manifest_path,manifest)
        for source in source_files:
            dest=output/'code_snapshot'/source.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    db=open_results(output/'results.sqlite')
    status={'state':'preparing','pid':os.getpid(),'started_utc':utcnow(),
            'execution':execution,
            'completed_fits':db.execute('SELECT count(*) FROM fits').fetchone()[0],
            'queued_stages':['BA+M5',f'BR all eligible series, {len(available_ufs)} available UFs']}
    write_json(output/'status.json',status)
    torch.set_num_threads(2)
    if os.name=='nt':
        import ctypes
        # Keep computation alive while allowing the display to switch off.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with ParallelFitPool(args.workers,args.device,priority=args.priority) as pool:
            if args.preflight:
                values=np.array([2,0,3,1,0,5,2,0,4,1,2,3,0,1,5,2,1]*2,float)[:31]
                run_parallel([('PREFLIGHT',{'origin':17,'uf':'TEST'},('TEST','synthetic','synthetic'),values)],pool,db,output,status)
            else:
                # Load only two cohorts concurrently; national states follow after both finish.
                ba=prepare('BA','BA',output);m5=prepare('M5','M5',output)
                national_plan=output/'national_plan.json'
                national=json.loads(national_plan.read_text(encoding='utf-8'))['cohorts'] if national_plan.exists() else []
                manifest['cohorts']=ba+m5+national;manifest['status']='running';write_json(manifest_path,manifest)
                status['planned_ba_m5_fits']=sum(m['eligible_series'] for m in ba+m5)*fits_per_case()
                print('BA+M5 census:',[(m['dataset'],m['origin'],m['eligible_series']) for m in ba+m5],flush=True)
                tagged=((dataset,*case) for pair in itertools.zip_longest(cases(ba,output),cases(m5,output))
                        for dataset,case in zip(['BA','M5'],pair) if case is not None)
                run_parallel(tagged,pool,db,output,status)
                status['queued_stages']=[f'BR all eligible series, {len(available_ufs)} available UFs']
                export_completed(db,output)
                from simulation.research.visualizations import generate_research_visualizations
                generate_research_visualizations(output)
                for uf in available_ufs:
                    status.update(state='preparing',dataset='BR',uf=uf,updated_utc=utcnow())
                    write_json(output/'status.json',status)
                    metadata=prepare('BR',uf,output)
                    manifest['cohorts']=[m for m in manifest['cohorts'] if not(m['dataset']=='BR' and m['uf']==uf)]+metadata
                    write_json(manifest_path,manifest)
                    print('National census:',uf,[(m['origin'],m['eligible_series']) for m in metadata],flush=True)
                    if uf=='BA':
                        # Exact same source, origins, code and budgets: reuse existing Bahia fits.
                        cursor=db.execute("SELECT task_id,origin,policy,seed,fit_seconds,payload FROM fits WHERE dataset='BA'")
                        while batch:=cursor.fetchmany(1000):
                          with db:
                            for tid,origin,policy,seed,seconds,payload in batch:
                                p=json.loads(payload);key=tuple(p['rows'][0][k] for k in KEYS)
                                for row in p['rows']:row['dataset']='BR';row['reused_from']='BA'
                                new_id=task_id('BR',uf,origin,key,policy,seed)
                                db.execute('INSERT OR IGNORE INTO fits VALUES (?,?,?,?,?,?,?,?)',
                                           [new_id,'BR',uf,origin,policy,seed,seconds,json.dumps(p)])
                        status['completed_fits']=db.execute('SELECT count(*) FROM fits').fetchone()[0]
                    run_parallel((('BR',*case) for case in cases(metadata,output)),pool,db,output,status)
                export_completed(db,output)
            status.update(state='completed',finished_utc=utcnow(),queued_stages=[])
            write_json(output/'status.json',status)
            manifest['status']='completed';manifest['finished_utc']=utcnow();write_json(manifest_path,manifest)
            if args.preflight:export_completed(db,output)
    except Exception:
        status.update(state='failed',updated_utc=utcnow(),error=traceback.format_exc())
        write_json(output/'status.json',status)
        manifest['status']='failed';write_json(manifest_path,manifest)
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        db.close();lock.close()


if __name__=='__main__':main()
