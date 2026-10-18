"""Operational elastic CPU launcher; frozen scientific code remains unchanged.

runtime_control.json sets workers (1..16). Decreases drain running fits;
increases spawn additional single-thread workers without restarting the queue.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[3]


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
        # Windows readers/antivirus can briefly prevent replacing an open file.
        for attempt in range(8):
            try:
                os.replace(temporary,path)
                return
            except PermissionError:
                if attempt==7:raise
                time.sleep(min(.02*2**attempt,.1))
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass


class ElasticFitPool:
    def __init__(self,output,workers,device='cpu',priority='below-normal'):
        if device!='cpu' or not 1<=workers<=16:raise ValueError('Expected CPU and 1..16 workers')
        self.output=Path(output);self.workers=workers;self.device=device;self.priority=priority
        self.slots=[];self.pending={};self.last_publish=0
        self.last_publish_warning=None

    def __enter__(self):return self

    def target(self):
        try:
            data=json.loads((self.output/'runtime_control.json').read_text(encoding='utf-8'))
            workers=data['workers']
            if type(workers) is int and 1<=workers<=16:self.workers=workers
        except (OSError,ValueError,KeyError):pass # transient/invalid request cannot stop training
        return self.workers

    def publish(self,state='running',force=False):
        now=time.monotonic()
        if not force and now-self.last_publish<1:return
        target=self.workers
        try:
            atomic_json(self.output/'runtime_pool.json',{
                'state':state,'pid':os.getpid(),'workers':target,
                'active_fits':len(self.pending),'worker_slots':len(self.slots),
                'retiring_workers':max(0,len(self.slots)-target),
                'device':'cpu','priority':self.priority,'threads_per_worker':1,
                'updated_utc':datetime.now(timezone.utc).isoformat()})
        except OSError as exc:
            # Monitoring is best effort. A locked status file must never discard
            # completed futures, stop training, or mask an actual training error.
            if self.last_publish_warning is None or now-self.last_publish_warning>=60:
                print(f'{datetime.now(timezone.utc).isoformat()} Monitoring update failed; '
                      f'training continues: {exc!r}',file=sys.stderr,flush=True)
                self.last_publish_warning=now
        finally:self.last_publish=time.monotonic()

    def new_slot(self):
        from simulation.research.parallel import initialize_worker
        return ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(self.device,None,self.priority))

    def results(self,jobs):
        from simulation.research.parallel import execute_fit
        iterator=iter(jobs);exhausted=False
        while self.pending or not exhausted:
            target=self.target()
            busy={slot for slot,job in self.pending.values()}
            # Only idle excess workers retire. Running fits always finish.
            for slot in self.slots[target:].copy():
                if slot not in busy:
                    slot.shutdown(wait=True);self.slots.remove(slot)
            while len(self.slots)<target and not exhausted:self.slots.append(self.new_slot())
            for slot in self.slots[:target]:
                if slot in busy or exhausted:continue
                try:job=next(iterator)
                except StopIteration:exhausted=True;break
                self.pending[slot.submit(execute_fit,job)]=(slot,job)
            self.publish()
            if not self.pending:break
            done,_=wait(self.pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                slot,job=self.pending.pop(future)
                yield job,future.result()

    def __exit__(self,*exc):
        for slot in self.slots:slot.shutdown(wait=True,cancel_futures=True)
        self.slots.clear();self.pending.clear();self.publish(state='closed',force=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum() or not 1<=args.workers<=16:
        raise ValueError('Invalid run or workers')
    sys.path.insert(0,str(ROOT/'experiment/scripts/analysis'))
    import run_aipe_pipeline as pipeline
    output=pipeline.BASE/args.run
    plan=json.loads((output/'plan.json').read_text(encoding='utf-8'))
    for name,digest in plan['code_sha256'].items():
        if pipeline.queue.sha256(ROOT/name)!=digest:raise ValueError('Frozen scientific code changed: '+name)
    execution=plan['execution']|{'workers':args.workers,'dynamic_workers':True}
    atomic_json(output/'runtime_control.json',{'workers':args.workers})
    atomic_json(output/'elastic_execution.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'original_plan_sha256':pipeline.queue.sha256(output/'plan.json'),
        'launcher_sha256':pipeline.queue.sha256(Path(__file__)),
        'original_execution':plan['execution'],'execution':execution})
    plan['execution']=execution
    # Replace only scheduling; execute_fit, task IDs, seeds, costs and scientific
    # source hashes are unchanged. The existing writer still owns all commits.
    pipeline.ParallelFitPool=lambda workers,device,priority:ElasticFitPool(output,args.workers,device,priority)
    pipeline.execute(output,plan)


if __name__=='__main__':main()
