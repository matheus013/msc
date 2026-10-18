"""Validate the bounded background profile after the CPU scaling screen.

Uses the same cases and seeds as benchmark_parallel_fits; never starts a census.
"""
import argparse
import json
from benchmark_parallel_fits import ROOT, jobs_for, trial, compare_results, write_json
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',required=True)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    for name in [args.run,args.reference]:
        if not name.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    reference=ROOT/'experiment/data/08_reporting/performance'/args.reference
    output=reference.parent/args.run;output.mkdir(exist_ok=False)
    full_reference=json.loads((reference/'full_cpu_1.json').read_text())['results']
    full=jobs_for(20000,[11,29]);short=jobs_for(2048,[11])
    summaries=[];checks=[]
    summary,results=trial('cpu',8,full,'full',output,priority='below-normal')
    summaries.append(summary)
    checks.append({'phase':'full','workers':8,'maximum_cpu_metric_difference':compare_results(full_reference,results)})
    baseline=None
    for device,workers in [('cpu',1),('cpu',8),('cuda',1),('cuda',2),('cuda',4)]:
        summary,results=trial(device,workers,short,'screen',output,priority='below-normal')
        summaries.append(summary)
        if device=='cpu':
            if baseline is None:baseline=results
            checks.append({'phase':'screen','workers':workers,'maximum_cpu_metric_difference':compare_results(baseline,results)})
        pd.DataFrame(summaries).to_csv(output/'throughput.csv',index=False)
    write_json(output/'manifest.json',{'status':'completed','priority':'below-normal',
        'reference':args.reference,'cpu_checks':checks,'production_queue':'not resumed',
        'scope':'One real Bahia series and one M5 series; DQN, PPO and GA-DQN. Full CPU: two seeds, 20,000 transitions; short CPU/CUDA: one seed, 2,048 transitions, rounded to episodes.',
        'limitations':['Concurrent user workload is uncontrolled; a single timing repetition per configuration.',
                       'Throughput includes evaluation; startup is separately reported. No census ETA or policy quality claim.']})
    write_json(output/'status.json',{'state':'completed'})


if __name__=='__main__':main()
