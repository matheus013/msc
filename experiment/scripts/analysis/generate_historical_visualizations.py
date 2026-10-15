"""Five requested diagnostics from preserved 12-policy matrices; no new fits."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.visualizations import KEYS,prefix_features,render_figures,pareto_mask


def main():
    source=ROOT/'experiment/data/08_reporting/sca_article/dual'
    output=source/'visualizations';output.mkdir(exist_ok=True)
    frontiers=[];selections=[];features=[];counts=[]
    for dataset,label,folder,origin in [('BA','real_BA','',17),('M5','M5','m5/',19)]:
        frame=pd.read_csv(source/f'{label}_policies.csv');frame['dataset']=dataset
        frame['pareto']=pareto_mask(frame.TIC,frame.NS);frontiers.append(frame)
        for alpha in [.7,.8,.85,.9,.95]:
            chosen=pd.read_csv(source/f'{label}_selections_{alpha:.2f}.csv',dtype={key:str for key in KEYS})
            chosen=chosen[chosen.strategy.isin(['global_store_out','profile_store_out'])].copy()
            chosen['dataset']=dataset;chosen['origin']=origin
            chosen['strategy']=chosen.strategy.map({'global_store_out':'global','profile_store_out':'perfil'})
            selections.append(chosen)
        scenarios=pd.read_parquet(ROOT/f'experiment/data/03_primary/{folder}scenarios.parquet')
        for key in KEYS+['venda_ciclo']:scenarios[key]=scenarios[key].astype(str)
        wide=scenarios.pivot(index=KEYS,columns='venda_ciclo',values='demand').sort_index(axis=1)
        if wide.isna().any().any():raise ValueError('Historical panel contains gaps; do not silently zero-fill')
        attributes=prefix_features(wide,origin);attributes['dataset']=dataset;attributes['origin']=origin
        features.append(attributes)
        counts.append({'dataset':dataset,'origin':origin,'candidate_series':len(wide),
                       'zero_positive_prefix':int(wide.iloc[:,:origin].sum(axis=1).eq(0).sum())})
    frontier=pd.concat(frontiers,ignore_index=True);chosen=pd.concat(selections,ignore_index=True)
    demand=pd.concat(features,ignore_index=True);prefix=pd.DataFrame(counts)
    figures=render_figures(frontier,chosen,demand,prefix,output,historical=True)
    frontier.to_csv(output/'frontier.csv',index=False);prefix.to_csv(output/'prefix_counts.csv',index=False)
    chosen.to_csv(output/'historical_store_out_selections.csv',index=False)
    demand.to_parquet(output/'historical_prefix_features.parquet',index=False)
    report={'status':'historical exploratory reanalysis; not corrected temporal validation',
        'policies':sorted(frontier.policy.unique()),'figures':figures,'sources':str(source.relative_to(ROOT)),
        'prefix_cycles':{'BA':17,'M5':19},'zero_prefix_counts':counts,
        'limitations':['Historical profiles use full histories; store exclusion alone does not establish temporal separation.',
                       'Old evaluation metrics retained as recorded; new reports preserve undefined fill rate for zero demand.',
                       'Demand panels describe the preserved historical cohorts, not the new eligible census.',
                       'Heatmap shows modal choices across store-out folds, not one universal policy per profile.',
                       'PNG and SVG exported; PDF backend unavailable because local application control blocks a fontTools binary.']}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
