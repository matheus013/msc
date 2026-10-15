"""Bounded process-level fitting with independent RNGs and a parent-only writer.

No worker writes SQLite or status files. Jobs contain training/validation/test
arrays only; process scheduling cannot change the task's seed or input data.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
import time


def initialize_worker(device, barrier=None, priority='normal'):
    if priority=='below-normal':
        import psutil
        process=psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 10)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    os.environ['AIPE_TORCH_THREADS']='1'
    os.environ['AIPE_DEVICE']=device
    if device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    from simulation.core.rl_torch import DoubleDQNAgent,PPOAgent
    from simulation.core.inventory_env import InventoryEnv
    from simulation.research.production import production_config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; benchmark must not silently fall back')
    train=np.array([1.,0.,3.,2.,0.,4.,2.,1.,0.,2.,4.,1.,0.,2.,3.,1.,2.])
    cfg=production_config(train,7)
    cfg['DQN']['episodes']=4;cfg['PPO']['episodes']=1
    for cls in [DoubleDQNAgent,PPOAgent]:
        agent=cls(InventoryEnv.STATE_DIM,cfg);agent.train(train,cfg,verbose=False)
    if device=='cuda':torch.cuda.synchronize()
    if barrier is not None:barrier.wait(timeout=180)


def worker_ready():
    return os.getpid()


def execute_fit(job):
    import math
    import numpy as np
    import torch
    import psutil
    from simulation.research.production import production_config,fit_production
    from simulation.research.protocol import common_initial_state,simulate,profile_of
    started=time.perf_counter()
    train=np.asarray(job['train'],float)
    validation=np.asarray(job['validation'],float);test=np.asarray(job['test'],float)
    cfg=production_config(train,job['seed'])
    transitions=job.get('transitions')
    if transitions is not None:
        for name in ['DQN','PPO']:cfg[name]['episodes']=math.ceil(transitions/len(train))
    # A worker is configured once. All sub-algorithms use the same device.
    device=os.environ['AIPE_DEVICE']
    cfg['SIMULATION']['device']=device
    for name in ['DQN','PPO']:cfg[name]['device']=device
    valid_cfg,test_cfg=common_initial_state(train,validation,cfg)
    if device=='cuda':torch.cuda.synchronize()
    before_fit=time.perf_counter()
    policy=fit_production(job['policy'],train,cfg)
    if device=='cuda':torch.cuda.synchronize()
    fitted=time.perf_counter()
    rows=[]
    for block,values,episode_cfg in [('validation',validation,valid_cfg),('test',test,test_cfg)]:
        k,_=simulate(values,episode_cfg,policy)
        row={'block':block,'TIC':k['TIC'],'NS':k['ServiceLevel'],'TR':k['StockoutRate'],
             'BE':k['BullwhipEffect'],'FP':k['OrderFrequency'],'Demand':k['DemandTotal'],
             'Served':k['DemandServed'],'TerminalCommittedUnits':k['TerminalCommittedUnits']}
        if not np.isfinite(row['TIC']):raise ValueError('Nonfinite evaluation cost')
        row={key:None if isinstance(value,float) and not np.isfinite(value) else value for key,value in row.items()}
        rows.append(row)
    memory=psutil.Process().memory_full_info()
    return {'job_id':job['job_id'],'policy':job['policy'],'seed':job['seed'],
            'device':device,'origin':len(train),'fit_seconds':fitted-before_fit,
            'evaluation_seconds':time.perf_counter()-fitted,'total_seconds':time.perf_counter()-started,
            'worker_pid':os.getpid(),'worker_uss_mb':getattr(memory,'uss',memory.rss)/1024**2,
            'process_priority':psutil.Process().nice(),
            'threads':torch.get_num_threads(),'config':cfg,'profile':profile_of(train),'rows':rows}


class ParallelFitPool:
    """Spawn workers once, bound pending jobs, and expose completion per fit."""
    def __init__(self,workers,device='cpu',max_pending=None,priority='normal'):
        if workers<1:raise ValueError('workers must be positive')
        if device not in {'cpu','cuda'}:raise ValueError('device must be cpu or cuda')
        if priority not in {'normal','below-normal'}:raise ValueError('invalid priority')
        self.priority=priority
        self.workers=workers;self.device=device
        self.max_pending=max_pending or workers*2
        if self.max_pending<workers:raise ValueError('max_pending must cover workers')

    def __enter__(self):
        context=multiprocessing.get_context('spawn')
        self.executor=ProcessPoolExecutor(max_workers=self.workers,mp_context=context,
            initializer=initialize_worker,initargs=(self.device,context.Barrier(self.workers),self.priority))
        try:
            futures=[self.executor.submit(worker_ready) for _ in range(self.workers)]
            for future in futures:future.result(timeout=240)
        except BaseException:
            self.executor.shutdown(wait=False,cancel_futures=True)
            raise
        return self

    def results(self,jobs):
        iterator=iter(jobs);pending={};exhausted=False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending)<self.max_pending:
                    try:job=next(iterator)
                    except StopIteration:exhausted=True;break
                    pending[self.executor.submit(execute_fit,job)]=job
                if not pending:break
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    job=pending.pop(future)
                    yield job,future.result()
        except BaseException:
            for future in pending:future.cancel()
            raise

    def __exit__(self,*exc):
        self.executor.shutdown(wait=True,cancel_futures=True)
