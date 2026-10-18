"""Exercise resizing with real futures and lightweight thread executors."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

PATH=Path(__file__).resolve().parents[1]/'scripts/operations/run_aipe_elastic.py'
spec=importlib.util.spec_from_file_location('elastic',PATH)
elastic=importlib.util.module_from_spec(spec);spec.loader.exec_module(elastic)


def test_resize_preserves_every_job_and_drains_busy_workers(tmp_path,monkeypatch):
    import simulation.research.parallel as parallel
    monkeypatch.setattr(parallel,'execute_fit',lambda job:(time.sleep(.02),job)[1])
    pool=elastic.ElasticFitPool(tmp_path,2)
    created=[]
    def new_slot():
        slot=ThreadPoolExecutor(max_workers=1);created.append(slot);return slot
    monkeypatch.setattr(pool,'new_slot',new_slot)
    results=[];peaks=[];retiring=[]
    with pool:
        for job,result in pool.results(range(30)):
            assert job==result;results.append(result);peaks.append(len(pool.slots))
            if len(results)==2:elastic.atomic_json(tmp_path/'runtime_control.json',{'workers':4})
            if len(results)==6:elastic.atomic_json(tmp_path/'runtime_control.json',{'workers':1})
            retiring.append(max(0,len(pool.slots)-pool.workers))
    assert sorted(results)==list(range(30)) and max(peaks)==4
    assert max(retiring)>0 and pool.workers==1 and not pool.pending
    assert json.loads((tmp_path/'runtime_pool.json').read_text())['state']=='closed'


def test_invalid_control_keeps_previous_target(tmp_path):
    pool=elastic.ElasticFitPool(tmp_path,8)
    for value in [True,0,17,'12',None]:
        elastic.atomic_json(tmp_path/'runtime_control.json',{'workers':value})
        assert pool.target()==8
    (tmp_path/'runtime_control.json').write_text('{partial')
    assert pool.target()==8


def test_atomic_update_retries_windows_sharing_failure(tmp_path,monkeypatch):
    real_replace=elastic.os.replace;attempts=[]
    def replace(source,destination):
        attempts.append(1)
        if len(attempts)<3:raise PermissionError('file in use')
        real_replace(source,destination)
    monkeypatch.setattr(elastic.os,'replace',replace)
    monkeypatch.setattr(elastic.time,'sleep',lambda seconds:None)
    path=tmp_path/'runtime_pool.json'
    elastic.atomic_json(path,{'workers':12})
    assert len(attempts)==3 and json.loads(path.read_text())['workers']==12
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_monitoring_failure_does_not_lose_jobs(tmp_path,monkeypatch,capsys):
    import simulation.research.parallel as parallel
    monkeypatch.setattr(parallel,'execute_fit',lambda job:job)
    monkeypatch.setattr(elastic.os,'replace',lambda *args:(_ for _ in ()).throw(PermissionError('locked')))
    monkeypatch.setattr(elastic.time,'sleep',lambda seconds:None)
    previous=tmp_path/'runtime_pool.json';previous.write_text('{"workers":8}')
    pool=elastic.ElasticFitPool(tmp_path,2)
    monkeypatch.setattr(pool,'new_slot',lambda:ThreadPoolExecutor(max_workers=1))
    with pool:results=list(pool.results(range(20)))
    assert sorted(result for job,result in results)==list(range(20))
    assert not pool.pending and not pool.slots
    assert json.loads(previous.read_text())=={'workers':8}
    assert not list(tmp_path.glob('*.tmp'))
    assert capsys.readouterr().err.count('training continues')==1
