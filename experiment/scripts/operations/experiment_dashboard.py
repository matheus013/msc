"""Local monitor with bounded operational concurrency control."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import re
import sqlite3
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[3]
RUNS_ROOT=ROOT/'experiment/data/08_reporting/research_v2'
DEFAULT_RUNS=['aipe_ba145_unified_2026-10-15','aipe_m5_2408_unified_2026-10-15']
HTML=Path(__file__).with_name('dashboard')/'index.html'
FAILED={'failed','evaluation_failed','interrupted','stopped_by_user'}


def read_json(path):
    if not path.exists():return {}
    return json.loads(path.read_text(encoding='utf-8-sig'))


def database_counts(path,table='fits'):
    """mode=ro ensures opening a missing or unavailable DB cannot create it."""
    if not path.exists():return None
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.5) as db:
        db.execute('PRAGMA query_only=ON')
        if table=='plans':return db.execute('SELECT count(*) FROM plans').fetchone()[0]
        return {policy:int(count) for policy,count in db.execute('SELECT policy,count(*) FROM fits GROUP BY policy')}


def run_snapshot(folder):
    plan=read_json(folder/'plan.json')
    status=read_json(folder/'status.json')
    pool=read_json(folder/'runtime_pool.json')
    execution=status.get('execution',plan.get('execution',{})).copy()
    if pool.get('state')=='running':
        execution.update({k:pool[k] for k in ['workers','threads_per_worker'] if k in pool})
    state=status.get('state',plan.get('state','unknown'))
    errors=[]
    try:counts=database_counts(folder/'results.sqlite')
    except (sqlite3.Error,OSError) as exc:
        counts=None;errors.append('Leitura dos checkpoints indisponível: '+str(exc))
    imported=(folder/'results.sqlite').exists()
    unknown_count=imported and counts is None
    available=(counts or {}) if imported else plan.get('reuse_by_policy',{})
    rows=[]
    for policy in plan.get('policies',[]):
        reused=int(plan.get('reuse_by_policy',{}).get(policy,0))
        total=reused+int(plan.get('pending_by_policy',{}).get(policy,0))
        completed=min(int(available.get(policy,0)),total)
        rows.append({'name':policy,'total':total,'completed':completed,'reusable':reused,
            'remaining':max(0,total-completed),'percent':100*completed/total if total else 0})
    planned=int(plan.get('planned_fits',status.get('planned_fits',0)))
    completed=sum(r['completed'] for r in rows)
    if not rows:completed=int(status.get('completed_fits',0))
    selectors=all((folder/'selection'/n).exists() for n in ['summary.csv','case_metrics.csv','predictions_frozen_before_scoring.csv'])
    plots=(folder/'visualizations/manifest.json').exists()
    try:oracles=database_counts(folder/'selection/oracle_cache.sqlite','plans') or 0
    except (sqlite3.Error,OSError) as exc:oracles=0;errors.append('Leitura dos oráculos indisponível: '+str(exc))
    eta=None
    if plan.get('linear_fit_hours_at_8_workers') is not None:
        original=sum(plan.get('pending_by_policy',{}).values())
        # A planning projection, not a live throughput estimate.
        eta=float(plan['linear_fit_hours_at_8_workers'])*8/max(1,execution.get('workers',8))*max(0,planned-completed)/original if original else 0
    files=[folder/'status.json',folder/'selection/summary.csv',folder/'selection/oracle_cache.sqlite']
    modified=max((p.stat().st_mtime for p in files if p.exists()),default=None)
    alive=None
    if status.get('pid'):
        try:
            import psutil
            process=psutil.Process(status['pid'])
            alive=process.is_running() and folder.name in ' '.join(process.cmdline())
        except Exception:alive=False
    return {'id':folder.name,'dataset':plan.get('dataset',status.get('dataset','?')),'state':state,
        'error':status.get('error'),'read_errors':errors,'completed':completed,'planned':planned,
        'percent':100*completed/planned if planned else 0,'remaining':max(0,planned-completed),
        'coverage_source':'current_queue' if imported else 'audited_reuse',
        'reusable':int(plan.get('reusable_fits',0)),'policies':rows,
        'cohorts':plan.get('cohorts',[]),'execution':execution,'pool':pool,
        'requested_workers':read_json(folder/'runtime_control.json').get('workers',execution.get('workers',8)),
        'fit_hours_projection':eta,'last_activity':datetime.fromtimestamp(modified,timezone.utc).isoformat() if modified else None,
        'process_alive':alive,'last_policy':status.get('policy'),
        'stages':{'controllers':planned>0 and completed>=planned,'selectors':selectors,'visualizations':plots,
                  'oracle_cached':oracles,'oracle_total':int(plan.get('series_origin_cases',0)),
        'evaluation':state=='completed'},'unknown_checkpoint_count':unknown_count}


class Monitor:
    def __init__(self,base=RUNS_ROOT,runs=None):
        self.base=Path(base);self.runs=runs or DEFAULT_RUNS;self.lock=threading.Lock();self.cached=None;self.updated=0
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+',name) for name in self.runs):raise ValueError('Invalid run name')

    def set_workers(self,run,workers):
        if run not in self.runs or type(workers) is not int or not 1<=workers<=16:
            raise ValueError('Use uma execução conhecida e de 1 a 16 processos')
        with self.lock:
            folder=self.base/run;snapshot=run_snapshot(folder)
            if snapshot['state']!='running' or snapshot.get('process_alive') is not True or snapshot['pool'].get('state')!='running':
                raise ValueError('O controle está disponível somente para uma fila elástica ativa')
            temp=folder/('runtime_control.'+uuid.uuid4().hex+'.tmp')
            request={'workers':workers,'requested_utc':datetime.now(timezone.utc).isoformat()}
            temp.write_text(json.dumps(request),encoding='utf-8');os.replace(temp,folder/'runtime_control.json')
            with (folder/'runtime_control_history.jsonl').open('a',encoding='utf-8') as history:
                history.write(json.dumps(request)+'\n')
            self.cached=None
            return request

    def snapshot(self):
        with self.lock:
            if self.cached and time.monotonic()-self.updated<4:return self.cached
            runs=[]
            for name in self.runs:
                try:runs.append(run_snapshot(self.base/name))
                except (ValueError,OSError,KeyError) as exc:
                    runs.append({'id':name,'dataset':'?','state':'unavailable','error':str(exc),'read_errors':[]})
            machine={}
            try:
                import psutil
                # A tiny bounded sample works across the server's changing request threads.
                machine={'cpu_percent':psutil.cpu_percent(interval=.05),'memory_percent':psutil.virtual_memory().percent}
            except ImportError:pass
            blocked=any(r['state'] in FAILED or (r.get('process_alive') is False and r['state'] in ['running','controllers_completed']) for r in runs)
            planned=sum(r.get('planned',0) for r in runs);completed=sum(r.get('completed',0) for r in runs)
            self.cached={'updated_at':datetime.now(timezone.utc).isoformat(),'runs':runs,'machine':machine,
                'blocked':blocked,'completed':completed,'planned':planned,'percent':100*completed/planned if planned else 0,
                'read_only':False,'refresh_seconds':5,'coverage_unknown':any(r.get('unknown_checkpoint_count') or r['state']=='unavailable' for r in runs)}
            self.updated=time.monotonic();return self.cached


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            expected=f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host')!=expected or self.headers.get('Origin')!='http://'+expected:
                self.send_error(403);return
            if self.path!='/api/workers':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024 or self.headers.get('Content-Type')!='application/json':
                    self.send_error(400);return
                request=json.loads(self.rfile.read(length))
                response=monitor.set_workers(request['run'],request['workers'])
            except (ValueError,KeyError,TypeError) as exc:
                self.send_error(400,str(exc));return
            content=json.dumps(response).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)

        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path=='/api/status':
                content=json.dumps(monitor.snapshot(),ensure_ascii=False,allow_nan=False).encode('utf-8');kind='application/json; charset=utf-8'
            elif path in ['/','/index.html']:
                content=HTML.read_bytes();kind='text/html; charset=utf-8'
            elif path=='/favicon.ico':self.send_response(204);self.end_headers();return
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
        def log_message(self,*args):pass
    return Handler


def main():
    parser=argparse.ArgumentParser(description='Painel local AIPE com controle de concorrência')
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--runs',nargs='+',default=DEFAULT_RUNS)
    args=parser.parse_args();server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(Monitor(runs=args.runs)))
    print(f'Painel AIPE: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
