"""Regression checks for durability and the production search space."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from simulation.research.production import production_config, POLICIES, fits_per_case
from simulation.core.metaheuristics_torch import TorchGA

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/analysis/run_production_v2.py'
spec=importlib.util.spec_from_file_location('production_queue',SCRIPT)
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def test_search_limits_do_not_grow_when_same_history_is_repeated():
    short=np.array([1,0,3,2,0,4],float)
    long=np.tile(short,4)
    a=TorchGA(short,production_config(short,11),device='cpu')
    b=TorchGA(long,production_config(long,11),device='cpu')
    assert np.allclose(a.ub.numpy(),b.ub.numpy())
    assert a.ub[1]<2000


def test_selected_families_match_kedro_and_training_plan():
    import yaml
    from simulation.pipelines.inventory_simulation.nodes import POLICY_NAMES, _is_enabled
    config=SCRIPT.parents[2]/'conf/base/parameters/simulation.yml'
    params=yaml.safe_load(config.read_text(encoding='utf-8'))['simulation']
    selected={name for names in POLICY_NAMES.values() for name in names if _is_enabled(name,params)}
    aliases={'EOQCycle':'EOQ','sSCycle':'sS','NewsvendorHeuristic':'Newsvendor'}
    assert selected=={aliases.get(name,name) for name in POLICIES}
    assert not params['policies']['zabraoui'] and params['policies']['proposed_architecture']
    assert fits_per_case()==3+3*3==12
    assert fits_per_case(seeds=[11])==len(POLICIES)
    assert fits_per_case(['GA','DQN','PPO','SA','GA-DQN'],[11,29])==10


def test_atomic_fit_resume_and_csv_reuse(tmp_path):
    db=queue.open_results(tmp_path/'results.sqlite')
    row={'dataset':'BA','TIC':12.,'NS':.8,'reused_from':None}
    with db:db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?,?)',
                      ['abc','BA','BA',17,'GA',11,1.,json.dumps({'rows':[row]})])
    # A restarted process sees only committed fits, not a half-written result.
    with pytest.raises(RuntimeError):
        with db:
            db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?,?)',
                       ['interrupted','BA','BA',17,'GA',29,1.,'{}'])
            raise RuntimeError('simulate interruption')
    db.close();db=queue.open_results(tmp_path/'results.sqlite')
    assert db.execute('SELECT count(*) FROM fits').fetchone()[0]==1
    row['dataset']='BR';row['reused_from']='BA'
    with db:db.execute('INSERT INTO fits VALUES (?,?,?,?,?,?,?,?)',
                      ['reused','BR','BA',17,'GA',11,1.,json.dumps({'rows':[row]})])
    queue.export_completed(db,tmp_path)
    assert 'BR,12.0,0.8,BA' in (tmp_path/'kpis.csv').read_text()
    db.close()


def test_seed_dataset_and_origin_are_distinct_checkpoint_units():
    key=('BA','store','product')
    assert len({queue.task_id(dataset,'BA',origin,key,'GA',seed)
                for dataset in ['BA','BR'] for origin in [17,23] for seed in [11,29,47]})==12


def test_parallel_parent_commits_and_resume_skips_only_finished_jobs(tmp_path):
    db=queue.open_results(tmp_path/'results.sqlite')
    tagged=[('BA',{'uf':'BA','origin':17},('BA','shop','item'),np.arange(31,dtype=float))]
    jobs=queue.pending_jobs(tagged,db)
    first=next(jobs)
    assert first['train']==list(range(17))
    assert first['validation']==list(range(17,24))
    assert first['test']==list(range(24,31))
    result={'job_id':first['job_id'],'profile':'test','fit_seconds':1.,'config':{},
            'device':'cpu','worker_pid':123,'threads':1,'evaluation_seconds':.1,
            'total_seconds':1.1,'rows':[{'block':'test','TIC':12.,'NS':None}]}
    queue.commit_fit(db,first,result)
    restarted=list(queue.pending_jobs(tagged,db))
    assert len(restarted)==fits_per_case()-1
    assert first['job_id'] not in {job['job_id'] for job in restarted}
    payload=json.loads(db.execute('SELECT payload FROM fits').fetchone()[0])
    assert payload['rows'][0]['store_id']=='shop'
    assert payload['execution']['device']=='cpu'
    with pytest.raises(ValueError,match='mismatch'):
        queue.commit_fit(db,restarted[0],result)
    assert db.execute('SELECT count(*) FROM fits').fetchone()[0]==1
    db.close()


def test_parallel_queue_bounds_pending_work_without_consuming_all_jobs():
    from concurrent.futures import Future
    from simulation.research.parallel import ParallelFitPool
    consumed=[]
    def source():
        for n in range(100):
            consumed.append(n)
            yield {'job_id':n}
    class ImmediateExecutor:
        def submit(self,function,job):
            future=Future();future.set_result({'job_id':job['job_id']});return future
    pool=ParallelFitPool(2,max_pending=4);pool.executor=ImmediateExecutor()
    results=pool.results(source())
    job,result=next(results)
    assert job['job_id']==result['job_id'] and len(consumed)==4
    results.close()


def test_late_state_records_keep_the_national_calendar_and_all_sales(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    folder=tmp_path/'data/source/vendas/uf=AP';folder.mkdir(parents=True)
    pd.DataFrame({'revendedor_cod':['shop']*4,'produto_cod':['item']*4,
                  'venda_ciclo':[202401,202401,202402,202403],
                  'venda_qtd':['1','2','4','5']}).to_parquet(folder/'raw.parquet')
    cohorts=queue.read_real('AP',tmp_path)
    first_origin,first,meta=cohorts[0]
    second_origin,second,meta2=cohorts[1]
    assert first_origin==17 and first.empty and meta['eligible_series']==0
    assert second_origin==23 and len(second)==1 and meta2['eligible_series']==1
    assert second.columns[0]=='202301' and second.columns[17]=='202401'
    assert second.iloc[0,17]==3 and second.iloc[0,:17].sum()==0


def test_windows_reader_lock_does_not_abort_a_committed_fit(tmp_path,monkeypatch):
    actual_replace=queue.os.replace
    calls=[]
    def temporarily_locked(source,target):
        calls.append(True)
        if len(calls)<3:raise PermissionError('Reader temporarily holds status')
        actual_replace(source,target)
    monkeypatch.setattr(queue.os,'replace',temporarily_locked)
    monkeypatch.setattr(queue.time,'sleep',lambda delay:None)
    queue.write_json(tmp_path/'status.json',{'completed_fits':9})
    assert json.loads((tmp_path/'status.json').read_text())['completed_fits']==9
    assert len(calls)==3
    monkeypatch.setattr(queue.os,'replace',lambda source,target:(_ for _ in ()).throw(PermissionError()))
    queue.write_json(tmp_path/'status.json',{'completed_fits':10})
    fallback=list(tmp_path.glob('status_latest_*.json'))
    assert len(fallback)==1 and json.loads(fallback[0].read_text())['completed_fits']==10
