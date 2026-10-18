"""Reduced M5 comparison using frozen controller outcomes; no simulation reruns."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import shutil
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
import numpy as np
import pandas as pd
from simulation.research.pse import CASE, score_predictions
from simulation.research.pse_consequences import predict_consequences, MARGINS


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return 'always_reference' if value==float('inf') else None
    return value


def write(path,value):
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='m5_pse_consequences30_v2_2026-10-15')
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    source=ROOT/'experiment/data/08_reporting/research_v2/m5_pse_pilot30_v2_2026-10-15'
    out=source.parent/args.run
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (ImportError,AttributeError):pass
    inputs=['features.csv','kpis.csv','predictions_frozen_before_test.csv']
    protocol={'source':source.name,'scope':'30 series, 10 stores, 3 origins; development only',
        'controller_fits_repeated':0,'alpha':.7,'reference':'Global from other stores validation',
        'model':'Separate fixed ExtraTrees forests: 64 trees, depth 4, leaf 3, seed 20261015; log1p cost and service',
        'margin_grid':list(MARGINS),'selection':'Nested leave-one-store-out, same origin only; minimize inner cost subject to inner mean service >= reference',
        'guard':'predicted service >= .7 and >= predicted reference service; cost improvement exceeds selected margin',
        'limitations':['Previously inspected pilot: exploratory, not independent confirmation',
                      'Reference/profile use group mean service; consequence decision uses individual predicted service',
                      'No calibrated safety guarantee; no hyperparameter search beyond inner margin',
                      'Seeds averaged, origins dependent, only ten store clusters'],
        'input_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in inputs},
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'experiment/src/simulation/research/pse_consequences.py']}}
    write(out/'protocol_frozen_before_predictions.json',protocol)
    snapshot=out/'code_snapshot'
    snapshot.mkdir()
    for relative in protocol['code_sha256']:
        shutil.copy2(ROOT/relative,snapshot/Path(relative).name)
    features=pd.read_csv(source/'features.csv')
    validation=pd.read_csv(source/'kpis.csv').query('block == "validation"').copy()
    predictions,folds=predict_consequences(features,validation,
        lambda o,s,m:print(f'origin={o} held_store={s} margin={m}',flush=True))
    controls=pd.read_csv(source/'predictions_frozen_before_test.csv')
    predictions=pd.concat([controls,predictions],ignore_index=True)
    # Keep an individual-validation choice as a non-contextual transfer control.
    from simulation.research.pse import best_policy
    own=validation.groupby(CASE).apply(best_policy,include_groups=False).rename('policy').reset_index()
    own['strategy']='Escolha_validacao_individual'
    predictions=pd.concat([predictions,own],ignore_index=True)
    if predictions.duplicated(CASE+['strategy']).any():raise ValueError('Duplicate recommendations')
    counts=predictions.groupby('strategy').size()
    if not counts.eq(90).all():raise ValueError(f'Unexpected coverage: {counts}')
    predictions.to_csv(out/'predictions_frozen_before_test.csv',index=False)
    write(out/'inner_validation.json',folds)
    # Test is read for scoring only after all choices have been written.
    test=pd.read_csv(source/'kpis.csv').query('block == "test"').copy()
    scored=score_predictions(predictions,test)
    scored.to_csv(out/'case_metrics.csv',index=False)
    base=scored[scored.strategy.eq('Global')]
    summary=[];intervals=[];rng=np.random.default_rng(20261015)
    for strategy,g in scored.groupby('strategy'):
        pair=g.merge(base[CASE+['TIC','NS','policy']],on=CASE,suffixes=('','_ref'),validate='one_to_one')
        dc=pair.TIC-pair.TIC_ref;ds=pair.NS-pair.NS_ref
        row={'strategy':strategy,'cases':len(g),'mean_cost':float(g.TIC.mean()),
            'mean_service':float(g.NS.mean()),'defined_service':int(g.NS.notna().sum()),
            'feasible':int(g.feasible.sum()),'cost_change_percent':float(100*dc.mean()/pair.TIC_ref.mean()),
            'service_change_pp':float(100*ds.mean()),'different_policy':int(pair.policy.ne(pair.policy_ref).sum()),
            'higher_cost_cases':int(dc.gt(1e-6).sum()),'lower_service_cases':int(ds.lt(-1e-9).sum()),
            'new_service_failures':int((pair.NS_ref.ge(.7-1e-9)&pair.NS.lt(.7-1e-9)).sum()),
            'recovered_service_cases':int((pair.NS_ref.lt(.7-1e-9)&pair.NS.ge(.7-1e-9)).sum())}
        summary.append(row)
        for metric,delta in [('cost_difference',dc),('service_difference',ds)]:
            t=pair[CASE].copy();t['delta']=delta
            groups=t.groupby('store_id').delta.agg(['sum','count'])
            ix=rng.integers(0,len(groups),size=(5000,len(groups)))
            samples=groups['sum'].to_numpy()[ix].sum(axis=1)/groups['count'].to_numpy()[ix].sum(axis=1)
            intervals.append({'strategy':strategy,'metric':metric,'mean':float(delta.mean()),
                'ci95_low':float(np.quantile(samples,.025)),'ci95_high':float(np.quantile(samples,.975)),
                'stores':len(groups),'draws':5000,'conditional_exploratory':True})
    summary=pd.DataFrame(summary).sort_values('mean_cost')
    summary.to_csv(out/'summary.csv',index=False)
    write(out/'paired_intervals.json',intervals)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=summary[summary.strategy.isin(['Global','Perfil','AIPE_PSE','Classe_floresta_controle','Consequencia','Consequencia_recuo','Consequencia_sem_perfil'])]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for r in selected.itertuples():axes[0].scatter(r.mean_service*100,r.mean_cost,label=r.strategy,s=65)
    axes[0].axvline(70,color='gray',ls='--');axes[0].set(xlabel='Fill rate médio (%)',ylabel='Custo médio',title='Piloto exploratório: custo e atendimento')
    axes[0].legend(fontsize=8)
    axes[1].barh(selected.strategy,selected.feasible);axes[1].set(xlabel='Casos com fill rate ≥70% (86 definidos)',title='Atendimento por série e origem')
    for suffix in ('png','svg','pdf'):fig.savefig(out/('comparison.'+suffix),dpi=160)
    plt.close(fig)
    elapsed=time.perf_counter()-start
    write(out/'completion.json',{'state':'completed','elapsed_seconds':elapsed,'inventory_fits':0,
                                'recommendations':len(predictions),'outer_folds':len(folds)})
    print(summary.to_string(index=False),flush=True)
    print(f'Completed in {elapsed:.1f}s: {out}',flush=True)


if __name__=='__main__':main()
