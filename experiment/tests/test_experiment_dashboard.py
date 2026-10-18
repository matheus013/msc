import importlib.util
import json
from pathlib import Path
import sqlite3
import threading
import urllib.request
from http.server import ThreadingHTTPServer

PATH=Path(__file__).resolve().parents[1]/'scripts/operations/experiment_dashboard.py'
spec=importlib.util.spec_from_file_location('dashboard',PATH)
dashboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(dashboard)


def fixture_run(tmp_path):
    folder=tmp_path/'example';folder.mkdir()
    plan={'dataset':'BA','state':'planned_not_started','policies':['GA'],'planned_fits':6,
          'reuse_by_policy':{'GA':2},'pending_by_policy':{'GA':4},'reusable_fits':2,
          'linear_fit_hours_at_8_workers':1,'series_origin_cases':2}
    (folder/'plan.json').write_text(json.dumps(plan))
    return folder


def test_planned_reuse_is_distinguished_from_current_queue(tmp_path):
    folder=fixture_run(tmp_path);r=dashboard.run_snapshot(folder)
    assert r['completed']==2 and r['remaining']==4
    assert r['coverage_source']=='audited_reuse'
    assert not (folder/'results.sqlite').exists()


def test_complete_fits_do_not_hide_failed_evaluation(tmp_path):
    folder=fixture_run(tmp_path)
    with sqlite3.connect(folder/'results.sqlite') as db:
        db.execute('CREATE TABLE fits (policy TEXT)')
        db.executemany('INSERT INTO fits VALUES (?)',[('GA',)]*6)
    (folder/'status.json').write_text(json.dumps({'state':'evaluation_failed','error':'oracle failed'}))
    r=dashboard.Monitor(base=tmp_path,runs=['example']).snapshot()
    assert r['blocked'] and r['runs'][0]['percent']==100
    assert r['runs'][0]['stages']['controllers']
    assert not r['runs'][0]['stages']['evaluation']
    assert r['runs'][0]['error']=='oracle failed'


def test_corrupt_database_is_marked_unknown_not_reused_as_current_progress(tmp_path):
    folder=fixture_run(tmp_path);(folder/'results.sqlite').write_text('invalid sqlite')
    r=dashboard.Monitor(base=tmp_path,runs=['example']).snapshot()
    assert r['coverage_unknown'] and r['runs'][0]['read_errors']


def test_server_reads_status_and_never_serves_workspace_files(tmp_path):
    fixture_run(tmp_path)
    server=ThreadingHTTPServer(('127.0.0.1',0),dashboard.make_handler(dashboard.Monitor(base=tmp_path,runs=['example'])))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(base+'/api/status') as response:
            assert response.headers['Cache-Control']=='no-store'
            assert not json.load(response)['read_only']
        with urllib.request.urlopen(base+'/') as response:assert b'Progresso dos experimentos' in response.read()
        import pytest
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(base+'/../../README.md')
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_control_rejects_other_origins_and_invalid_workers(tmp_path,monkeypatch):
    folder=fixture_run(tmp_path)
    monkeypatch.setattr(dashboard,'run_snapshot',lambda p:{'state':'running','process_alive':True,'pool':{'state':'running'}})
    monitor=dashboard.Monitor(base=tmp_path,runs=['example'])
    server=ThreadingHTTPServer(('127.0.0.1',0),dashboard.make_handler(monitor))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(workers,origin):
        return urllib.request.urlopen(urllib.request.Request(base+'/api/workers',
            data=json.dumps({'run':'example','workers':workers}).encode(),
            headers={'Content-Type':'application/json','Origin':origin}))
    import pytest
    try:
        with pytest.raises(urllib.error.HTTPError) as error:post(12,'https://example.com')
        assert error.value.code==403 and not (folder/'runtime_control.json').exists()
        with pytest.raises(urllib.error.HTTPError):post(17,base)
        with post(12,base) as response:assert json.load(response)['workers']==12
        assert json.loads((folder/'runtime_control.json').read_text())['workers']==12
        with pytest.raises(ValueError):monitor.set_workers('../example',12)
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
