"""Exploratory pruning on historical matrices; never touches experiment configs."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiment/data/08_reporting/performance/historical_pruning_2026-10-14'
KEYS=['warehouse','store_id','item_id']
PORTFOLIOS={'six':['EOQ','VendorResponsive','FixedInterval','CappedBaseStock','SA','PPO'],
            'reviewed_six':['EOQ','sS','CappedBaseStock','GA','PPO','GA-DQN'],
            'representative_five':['EOQ','Newsvendor','CappedBaseStock','GA','PPO'],
            'seven_hybrid':['EOQ','VendorResponsive','FixedInterval','CappedBaseStock','SA','PPO','GA-DQN'],
            'eleven_without_dqn':['EOQ','sS','Newsvendor','BigDataNewsvendor','VendorResponsive','FixedInterval','CappedBaseStock','GA','SA','PPO','GA-DQN'],
            'ten_without_dqn_ss':['EOQ','Newsvendor','BigDataNewsvendor','VendorResponsive','FixedInterval','CappedBaseStock','GA','SA','PPO','GA-DQN'],
            'nine_without_dqn_ga_ss':['EOQ','Newsvendor','BigDataNewsvendor','VendorResponsive','FixedInterval','CappedBaseStock','SA','PPO','GA-DQN']}


def frontier(frame):
    return {name:[other for other,b in frame.iterrows() if other!=name and
                 b.TIC<=a.TIC and b.NS>=a.NS and (b.TIC<a.TIC or b.NS>a.NS)]
            for name,a in frame.iterrows()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=[];niches=[];portfolios=[];redundancy=[];provenance={}
    for dataset,folder in [('BA',''),('M5','m5/')]:
        path=ROOT/f'experiment/data/07_model_output/{folder}kpis.parquet'
        profiles=ROOT/f'experiment/data/04_feature/{folder}demand_profiles.parquet'
        k=pd.read_parquet(path);p=pd.read_parquet(profiles)
        for f in [k,p]:
            for key in KEYS:f[key]=f[key].astype(str)
        assert not k.duplicated(KEYS+['policy']).any()
        k=k.merge(p[KEYS+['operational_profile']],on=KEYS,validate='many_to_one')
        means=k.groupby('policy')[['TIC','NS']].mean()
        dominated=frontier(means)
        for name,row in means.iterrows():
            summary.append({'dataset':dataset,'policy':name,'mean_TIC':row.TIC,'mean_NS':row.NS,
                'mean_cost_service_dominators':'; '.join(dominated[name])})
        a=k[k.policy.eq('EOQ')].set_index(KEYS).sort_index()
        b=k[k.policy.eq('sS')].set_index(KEYS).sort_index()
        redundancy.append({'dataset':dataset,'pair':'EOQ / sS',
            'cost_spearman':a.TIC.corr(b.TIC,method='spearman'),
            'same_cost_and_service_fraction':float((np.isclose(a.TIC,b.TIC,rtol=0,atol=1e-6)&np.isclose(a.NS,b.NS,rtol=0,atol=1e-9)).mean()),
            'median_absolute_cost_difference_fraction':float((abs(a.TIC-b.TIC)/a.TIC.clip(lower=1e-9)).median())})
        scenarios=pd.read_parquet(ROOT/f'experiment/data/03_primary/{folder}scenarios.parquet')
        for key in KEYS:scenarios[key]=scenarios[key].astype(str)
        cutoff=17 if dataset=='BA' else 19
        maturity={key:int((g.sort_values('venda_ciclo').demand.to_numpy(float)[:cutoff]>0).sum())
                  for key,g in scenarios.groupby(KEYS)}
        mature=[key for key in a.index if maturity[key]>=3]
        am=a.loc[mature];bm=b.loc[mature]
        redundancy[-1].update(mature_prefix_series=len(mature),
            mature_cost_spearman=am.TIC.corr(bm.TIC,method='spearman'),
            mature_same_cost_service_fraction=float((np.isclose(am.TIC,bm.TIC,rtol=0,atol=1e-6)&np.isclose(am.NS,bm.NS,rtol=0,atol=1e-9)).mean()))
        for alpha in [.7,.9,.95]:
            for profile,rows in k.groupby('operational_profile'):
                m=rows.groupby('policy')[['TIC','NS']].mean()
                feasible=m[m.NS>=alpha].sort_values(['TIC','NS'],ascending=[True,False])
                winner=feasible.index[0] if len(feasible) else None
                niches.append({'dataset':dataset,'profile':profile,'alpha':alpha,'series':rows[KEYS].drop_duplicates().shape[0],
                               'winner':winner,'TIC':None if winner is None else m.loc[winner,'TIC'],
                               'NS':None if winner is None else m.loc[winner,'NS']})
            # Retrospective per-series oracle: sensitivity diagnostic, not deployable performance.
            for name,policies in PORTFOLIOS.items():
                full=k[k.NS>=alpha].copy();reduced=full[full.policy.isin(policies)]
                minimum=full.groupby(KEYS).TIC.min()
                reduced_min=reduced.groupby(KEYS).TIC.min()
                paired=pd.concat([minimum.rename('full'),reduced_min.rename('reduced')],axis=1).dropna()
                full_feasible=len(minimum);feasibility_losses=full_feasible-len(paired)
                best=full.merge(minimum.rename('minimum'),on=KEYS)
                tied=best[np.isclose(best.TIC,best.minimum,rtol=0,atol=1e-6)].copy()
                tied['n_tied']=tied.groupby(KEYS).policy.transform('size')
                unique_removed_wins=tied[tied.n_tied.eq(1)&~tied.policy.isin(policies)].policy.value_counts().to_dict()
                totals=paired[['full','reduced']].sum().to_numpy()
                portfolios.append({'dataset':dataset,'portfolio':name,'alpha':alpha,
                    'full_feasible_series':full_feasible,'lost_feasible_series':feasibility_losses,
                    'oracle_cost_increase_percent_on_common_feasible':100*(totals[1]/totals[0]-1) if totals[0] else None,
                    'unique_removed_winners':json.dumps(unique_removed_wins)})
        provenance[dataset]={'kpis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'profiles_sha256':hashlib.sha256(profiles.read_bytes()).hexdigest(),
                             'series':len(p)}
    for name,rows in [('policy_summary',summary),('profile_winners',niches),('portfolio_sensitivity',portfolios),('redundancy',redundancy)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
        print(name,flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)
    (OUT/'manifest.json').write_text(json.dumps({'status':'exploratory historical screening, not confirmatory pruning',
        'sources':provenance,'candidate_portfolios':PORTFOLIOS,
        'limitations':['Historical preprocessing, initialization, repetitions and M5 cold-start problems prevent causal algorithm rankings.',
            'Global dominance in two means does not imply dominance at individual series or profiles.',
            'Oracle sensitivity uses realized historical outcomes and must not be presented as future test performance.',
            'Correlation of KPI costs does not prove identical order trajectories.',
            'GA and DQN are required ablations if a GA-DQN contribution is claimed.']},indent=2),encoding='utf-8')


if __name__=='__main__':main()
