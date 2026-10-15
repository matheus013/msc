"""CPU/CUDA process-throughput validation; isolated from the stopped census."""
from pathlib import Path
import os
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
os.environ['AIPE_TORCH_THREADS']='1'
import argparse
import hashlib
import json
import sys
import time
import threading
import psutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.parallel import ParallelFitPool


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def jobs_for(transitions,seeds):
    panel=ROOT/'experiment/data/08_reporting/research_v2/census_2026-10-14/panels'
    rows=[]
    for dataset,file,origin in [('BA','BA_BA_17.parquet',17),('M5','M5_M5_50.parquet',50)]:
        wide=pd.read_parquet(panel/file)
        for seed in seeds:
            # Odd seeds select the same single series per base; seeds remain task-specific.
            values=wide.iloc[(seed%2),:].to_numpy(float)
            for policy in ['DQN','PPO','GA-DQN']:
                rows.append({'job_id':f'{dataset}_{policy}_{seed}_{transitions}',
                    'dataset':dataset,'policy':policy,'seed':seed,'transitions':transitions,
                    'train':values[:origin].tolist(),'validation':values[origin:origin+7].tolist(),
                    'test':values[origin+7:origin+14].tolist()})
    return rows


def trial(device,workers,jobs,label,output,priority='normal'):
    start=time.perf_counter();measure={'max_cpu_percent':0.,'min_available_ram_mb':float('inf')}
    stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            measure['max_cpu_percent']=max(measure['max_cpu_percent'],psutil.cpu_percent())
            measure['min_available_ram_mb']=min(measure['min_available_ram_mb'],psutil.virtual_memory().available/1024**2)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    results=[]
    try:
        with ParallelFitPool(workers,device,priority=priority) as pool:
            ready=time.perf_counter()
            for job,result in pool.results(jobs):
                results.append(result|{'dataset':job['dataset']})
                write_json(output/'status.json',{'state':'benchmarking','phase':label,'device':device,
                    'workers':workers,'completed':len(results),'jobs':len(jobs),'elapsed_seconds':time.perf_counter()-ready})
                print(f'{label} {device} workers={workers} completed={len(results)}/{len(jobs)}',flush=True)
            end=time.perf_counter()
    finally:
        stop.set();watcher.join(timeout=2)
    summary={'phase':label,'device':device,'workers':workers,'priority':priority,'jobs':len(jobs),
        'startup_seconds':ready-start,'steady_seconds':end-ready,
        'wall_seconds_with_shutdown':time.perf_counter()-start,
        'fits_per_minute':60*len(jobs)/(end-ready),
        'mean_fit_seconds':sum(r['fit_seconds'] for r in results)/len(results),
        'max_worker_uss_mb':max(r['worker_uss_mb'] for r in results),**measure}
    write_json(output/(f'{label}_{device}_{workers}.json'),{'summary':summary,'results':results})
    print(json.dumps(summary),flush=True)
    return summary,results


def compare_results(baseline,other):
    expected={r['job_id']:r for r in baseline}
    maximum=0.
    for result in other:
        reference=expected[result['job_id']]
        for a,b in zip(reference['rows'],result['rows']):
            for metric in ['TIC','NS','TR','BE','FP','Demand','Served','TerminalCommittedUnits']:
                if a[metric] is None or b[metric] is None:
                    if a[metric]!=b[metric]:raise AssertionError('Changed undefined metric')
                else:maximum=max(maximum,abs(a[metric]-b[metric]))
    if maximum>1e-8:raise AssertionError(f'CPU scheduling changed results by {maximum}')
    return maximum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--workers',nargs='+',type=int,default=[1,2,4,8,12,16])
    parser.add_argument('--screen-transitions',type=int,default=2048)
    parser.add_argument('--full-transitions',type=int,default=20000)
    parser.add_argument('--gpu-workers',nargs='+',type=int,default=[1,2,4])
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    output=ROOT/'experiment/data/08_reporting/performance'/args.run
    output.mkdir(parents=True,exist_ok=False)
    screen=jobs_for(args.screen_transitions,[11,29,47,71]);full=jobs_for(args.full_transitions,[11,29])
    summaries=[];checks=[];baseline=None
    for workers in args.workers:
        if workers>psutil.cpu_count():continue
        if summaries:
            private=max(r['max_worker_uss_mb'] for r in summaries)
            if workers*private>psutil.virtual_memory().available/1024**2*.75:
                print(f'Skipping {workers}: memory budget',flush=True);continue
        summary,results=trial('cpu',workers,screen,'screen',output)
        summaries.append(summary)
        if baseline is None:baseline=results
        checks.append({'workers':workers,'max_absolute_cpu_metric_difference':compare_results(baseline,results)})
        pd.DataFrame(summaries).to_csv(output/'throughput.csv',index=False)
    best=max(summaries,key=lambda r:r['fits_per_minute'])['workers']
    # Verify long production budgets, including hybrid fit, not only network microkernels.
    for workers in sorted({1,min(best,len(full))}):
        summary,results=trial('cpu',workers,full,'full',output);summaries.append(summary)
        if workers==1:full_baseline=results
        checks.append({'full_workers':workers,'max_absolute_cpu_metric_difference':compare_results(full_baseline,results)})
        pd.DataFrame(summaries).to_csv(output/'throughput.csv',index=False)
    # Same short workloads on CUDA; evaluate multi-process GPU contention too.
    for workers in args.gpu_workers:
        summary,_=trial('cuda',workers,screen,'screen',output);summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output/'throughput.csv',index=False)
    # Full-budget fresh matched CUDA controls (DQN, PPO, hybrid; no extrapolation).
    gpu_full=[j for j in full if j['seed']==11]
    summary,_=trial('cuda',1,gpu_full,'full',output);summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output/'throughput.csv',index=False)
    write_json(output/'manifest.json',{'status':'completed isolated benchmark, not policy efficacy evidence',
        'cpu_logical_processors':psutil.cpu_count(),'cpu_physical_cores':psutil.cpu_count(logical=False),
        'best_cpu_screen_workers':best,'cpu_equivalence_checks':checks,
        'methods':['DQN','PPO','GA-DQN'],'screen_jobs':len(screen),'full_cpu_jobs':len(full),
        'per_worker_torch_threads':1,'production_queue':'not resumed',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['One series per base, two training origins only; concurrent user workload is uncontrolled.',
            'CPU scheduling equivalence is checked; CUDA RNG streams can yield different learned policies.',
            'Cold process startup reported separately; short screen throughput is not a full-census ETA.']})
    write_json(output/'status.json',{'state':'completed','best_cpu_screen_workers':best})


if __name__=='__main__':main()
