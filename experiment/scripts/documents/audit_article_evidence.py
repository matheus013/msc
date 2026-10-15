"""Independent, read-only audit of the Bahia article's numerical evidence.

Recomputes held-store choices by explicit filtering rather than the optimized
sufficient-statistics implementation. No fitting or experiment files changed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiment/data/08_reporting/research_v2/bahia145_cpu16_2026-10-15'
OUT=ROOT/'docs/supply_chain_analytics_article/reviews/fine_review_2026-10-15'
OUT.mkdir(exist_ok=True,parents=True)
manifest=json.loads((RUN/'manifest.json').read_text())
assert manifest['status']=='completed'
dependencies=['experiment/src/simulation/core/policies_sota.py',
              'experiment/src/simulation/core/inventory_env_torch.py',
              'experiment/src/simulation/core/metaheuristics_torch.py',
              'experiment/src/simulation/research/protocol.py',
              'experiment/src/simulation/research/production.py',
              'experiment/src/simulation/research/hybrid.py']
frozen_hashes={name.replace('\\','/'):digest for name,digest in manifest['code_sha256'].items()}
for name in dependencies:
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==frozen_hashes[name],name
keys=['dataset','origin','warehouse','store_id','item_id']
raw=pd.read_csv(RUN/'kpis.csv',dtype={c:str for c in ['warehouse','store_id','item_id']})
assert len(raw)==8700 and not raw.duplicated(keys+['policy','seed','block']).any()
assert raw.groupby(keys+['block']).Demand.nunique().eq(1).all()
for (_,g) in raw.groupby(keys,observed=True):
    expected={(p,s,b) for p in manifest['policies']
              for s in (manifest['independent_training_seeds'] if p in ['GA','PPO','DQN','GA-DQN'] else [0])
              for b in ['validation','test']}
    assert set(zip(g.policy,g.seed,g.block))==expected
assert np.allclose(raw.loc[raw.Demand.gt(0),'NS'],raw.loc[raw.Demand.gt(0),'Served']/raw.loc[raw.Demand.gt(0),'Demand'])
assert raw.loc[raw.Demand.eq(0),'NS'].isna().all()
avg=raw.groupby(keys+['profile','policy','block'],observed=True)[['TIC','NS','Demand','Served']].mean().reset_index()

def choice(g,alpha):
    means=g.groupby('policy',observed=True)[['TIC','NS']].mean().dropna()
    feasible=means[means.NS.ge(alpha)]
    if feasible.empty:
        return means.sort_values(['NS','TIC','policy'],ascending=[False,True,True]).index[0],True
    return feasible.sort_values(['TIC','NS','policy'],ascending=[True,False,True]).index[0],False

pieces=[]
for origin,cohort in avg.groupby('origin',observed=True):
    valid=cohort[cohort.block.eq('validation')]
    test=cohort[cohort.block.eq('test')]
    for (warehouse,store),held in test.groupby(['warehouse','store_id'],observed=True):
        other=valid[~(valid.warehouse.eq(warehouse)&valid.store_id.eq(store))]
        for alpha in [.7,.8,.85,.9,.95]:
            gp,gf=choice(other,alpha)
            for profile,group in held.groupby('profile',observed=True):
                vp=other[other.profile.eq(profile)]
                pp,pf=choice(vp,alpha) if vp.NS.notna().any() else (gp,True)
                for strategy,p,f in [('global',gp,gf),('perfil',pp,pf)]:
                    chosen=group[group.policy.eq(p)].copy()
                    chosen['strategy']=strategy;chosen['alpha']=alpha;chosen['fallback']=f
                    pieces.append(chosen)
sel=pd.concat(pieces,ignore_index=True)
original=pd.read_csv(ROOT/'docs/scientific_writing_review_2026-10-15/bahia_completed/selections.csv',
                     dtype={c:str for c in ['warehouse','store_id','item_id']})
identity=keys+['alpha','strategy']
joined=sel.merge(original,on=identity,suffixes=('_audit','_reported'),validate='one_to_one')
assert len(joined)==2900
assert joined.policy_audit.eq(joined.policy_reported).all()
assert joined.fallback_audit.eq(joined.fallback_reported).all()
assert np.allclose(joined.TIC_audit,joined.TIC_reported)
assert np.allclose(joined.NS_audit,joined.NS_reported,equal_nan=True)

# Bootstrap cluster sums and denominators, keeping the pair and all origins.
selected=sel[np.isclose(sel.alpha,.7)]
cost=selected.pivot(index=keys,columns='strategy',values='TIC').reset_index()
service=selected.pivot(index=keys,columns='strategy',values='NS').reset_index()
assert service['global'].isna().equals(service['perfil'].isna())
groups=['warehouse','store_id']
stores=cost[groups].drop_duplicates().sort_values(groups)
ci=cost.groupby(groups)[['global','perfil']].sum().reindex(pd.MultiIndex.from_frame(stores))
si=service.groupby(groups)[['global','perfil']].sum().reindex(ci.index)
ni=service.groupby(groups)['global'].count().reindex(ci.index).to_numpy()
rng=np.random.default_rng(20261015)
w=rng.multinomial(len(stores),np.full(len(stores),1/len(stores)),size=5000)
c=w@ci.to_numpy();s=w@si.to_numpy();n=w@ni
cost_inc=100*(c[:,1]-c[:,0])/c[:,0]
service_pp=100*(s[:,1]-s[:,0])/n
report={
    'audit':'independent explicit-filter selection recomputation',
    'scientific_dependencies_match_run_manifest':dependencies,
    'matched_decisions':len(joined),'fits':len(raw.drop_duplicates(keys+['policy','seed'])),
    'test_cases':290,'defined_test_service':283,'undefined_test_service':7,
    'zero_demand_validation_cases':int(avg[avg.block.eq('validation')].drop_duplicates(keys).Demand.eq(0).sum()),
    'cost_increase_percent':100*(cost.perfil.mean()-cost['global'].mean())/cost['global'].mean(),
    'cost_increase_CI95_conditional':np.quantile(cost_inc,[.025,.975]).tolist(),
    'service_change_pp':100*(service.perfil.mean()-service['global'].mean()),
    'service_change_CI95_conditional_pp':np.quantile(service_pp,[.025,.975]).tolist(),
    'bootstrap':{'draws':5000,'seed':20261015,'clusters':'115 stores; fitted controllers and choices fixed'},
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [RUN/'kpis.csv',RUN/'manifest.json']}}
summary=[]
for (alpha,strategy),g in sel.groupby(['alpha','strategy'],observed=True):
    summary.append({'alpha':alpha,'strategy':strategy,'cost':g.TIC.mean(),
                    'service':g.NS.mean(),'weighted_service':g.Served.sum()/g.Demand.sum(),
                    'fallback_cases':int(g.fallback.sum()),'below_070':int(g.NS.lt(.7).sum())})
report['threshold_sensitivity']=summary
(OUT/'numerical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','threshold_sensitivity']},indent=2))
