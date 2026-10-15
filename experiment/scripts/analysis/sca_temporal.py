"""Fixed calibration/validation/test diagnostic; see docs temporal_protocol.md."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiment/src'))
from simulation.core.inventory_env import InventoryEnv
from simulation.core.policies import EOQPolicy, SsPolicyClass, NewsvendorPolicy
from simulation.core.policies_zabraoui import FixedIntervalPolicy, VendorResponsivePolicy
from simulation.pipelines.demand_profiling.nodes import _series_features, classify_operational_profiles
from sca_validation import choose

OUT = ROOT / 'experiment/data/08_reporting/sca_article/temporal'
KEYS = ['warehouse', 'store_id', 'item_id']
POLICIES = {'EOQ': EOQPolicy, 'sS': SsPolicyClass, 'Newsvendor': NewsvendorPolicy,
            'FixedInterval': FixedIntervalPolicy, 'VendorResponsive': VendorResponsivePolicy}


class EOQPerCycle(EOQPolicy):
    """Audit sensitivity only: align demand and holding-cost time units."""
    def __init__(self, demand, cfg):
        super().__init__(demand, cfg)
        self.Q = max(1., np.sqrt(2*np.mean(demand)*cfg['COST']['ordering_cost_per_order']/cfg['COST']['holding_cost_per_unit']))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'experiment/data/03_primary/scenarios.parquet'
    frame = pd.read_parquet(source)
    records, profiles = [], []
    for key, rows in frame.sort_values(KEYS + ['venda_ciclo']).groupby(KEYS):
        d = rows.demand.to_numpy(float)
        if len(d) < 33:
            continue
        train = d[:17]
        feature = dict(zip(KEYS, key)) | _series_features(train, 17)
        profiles.append(feature)
        cfg = {'SIMULATION': {'lead_time': 2, 'initial_inventory': float(train.mean()*2 + 1.28*train.std()*np.sqrt(2))},
               'COST': {'holding_cost_per_unit': 1., 'stockout_cost_per_unit': 5.,
                        'ordering_cost_per_order': 50., 'ordering_cost_per_unit': .5},
               'HEURISTIC': {'z_score': 1.28}}
        for name, cls in (POLICIES | {'EOQPerCycle': EOQPerCycle}).items():
            with contextlib.redirect_stdout(io.StringIO()):
                policy = cls(train, cfg)
            for block, demand in [('validation', d[17:27]), ('test', d[27:])]:
                env = InventoryEnv(demand, cfg)
                # Policies use inventory/pipeline/past fill only, never state normalizers.
                state = env.reset()
                done = False
                while not done:
                    state, _, done, _ = env.step(policy(state, env))
                k = env.kpis()
                records.append(dict(zip(KEYS, key)) | {'policy': name, 'block': block,
                     'TIC': k['TIC'], 'NS_raw': k['ServiceLevel'],
                     'NS': k['ServiceLevel'] if demand.sum()>0 else 1., 'demand': float(demand.sum()),
                     'served': env._demand_served, 'T': len(demand)})
    profiles = classify_operational_profiles(pd.DataFrame(profiles), {})
    kpis = pd.DataFrame(records).merge(profiles[KEYS+['operational_profile']], on=KEYS, validate='many_to_one')
    assert not kpis.duplicated(KEYS+['policy','block']).any()
    kpis.to_csv(OUT / 'kpis.csv', index=False)
    profiles.to_csv(OUT / 'calibration_profiles.csv', index=False)
    val_all, test_all = [kpis[kpis.block.eq(b)] for b in ['validation', 'test']]
    val, test = [r[r.policy.isin(POLICIES)] for r in [val_all, test_all]]
    selections, summary, contrasts = [], [], []
    for alpha in [.7, .8, .85, .9, .95]:
        selected = []
        for _, row in profiles.iterrows():
            train_rows = val[~val.store_id.eq(row.store_id)]
            group = train_rows[train_rows.operational_profile.eq(row.operational_profile)]
            global_policy, global_fallback = choose(train_rows, alpha)
            policy, group_fallback = choose(group if len(group) else train_rows, alpha)
            held = test[np.logical_and.reduce([test[k].eq(row[k]) for k in KEYS])].set_index('policy')
            record = {k: row[k] for k in KEYS + ['operational_profile']} | {'alpha': alpha,
                      'global_fallback': global_fallback, 'profile_fallback': group_fallback,
                      'unseen_profile': len(group) == 0}
            for strategy, name in [('global',global_policy),('profile',policy),('EOQ','EOQ')]:
                record[strategy+'_policy'] = name
                for metric in ['TIC','NS','demand','served']:
                    record[strategy+'_'+metric] = float(held.loc[name,metric])
            corrected = test_all[np.logical_and.reduce([test_all[k].eq(row[k]) for k in KEYS]) & test_all.policy.eq('EOQPerCycle')].iloc[0]
            for metric in ['TIC','NS','demand','served']:
                record['EOQPerCycle_'+metric] = float(corrected[metric])
            selected.append(record)
        selected = pd.DataFrame(selected)
        selections.extend(selected.to_dict('records'))
        for strategy in ['global','profile','EOQ','EOQPerCycle']:
            summary.append({'alpha':alpha, 'strategy':strategy,
                'TIC_mean':selected[strategy+'_TIC'].mean(), 'NS_mean':selected[strategy+'_NS'].mean(),
                'demand_weighted_NS':selected[strategy+'_served'].sum()/selected[strategy+'_demand'].sum(),
                'series_below_alpha':int((selected[strategy+'_NS']<alpha).sum()),
                'NS_mean_positive_demand_only': selected.loc[selected[strategy+'_demand']>0,strategy+'_NS'].mean(),
                'policy_counts':selected[strategy+'_policy'].value_counts().to_dict() if strategy!='EOQPerCycle' else {'EOQPerCycle':len(selected)}})
        rng = np.random.default_rng(20261014)
        stores = selected.store_id.unique()
        groups = {s:np.flatnonzero(selected.store_id.eq(s)) for s in stores}
        for baseline in ['global','EOQ','EOQPerCycle']:
            a, b = selected[baseline+'_TIC'].to_numpy(), selected.profile_TIC.to_numpy()
            service = (selected.profile_NS-selected[baseline+'_NS']).to_numpy()
            samples=[]
            for _ in range(5000):
                idx=np.concatenate([groups[s] for s in rng.choice(stores,len(stores),replace=True)])
                samples.append([100*(a[idx].sum()-b[idx].sum())/a[idx].sum(),service[idx].mean()])
            ci=np.quantile(samples,[.025,.975],axis=0)
            contrasts.append({'alpha':alpha, 'baseline':baseline, 'treatment':'profile',
                'cost_reduction_percent':100*(a.sum()-b.sum())/a.sum(),
                'cost_reduction_ci':ci[:,0].tolist(), 'service_difference':service.mean(),
                'service_difference_ci':ci[:,1].tolist()})
    pd.DataFrame(selections).to_csv(OUT/'selections.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    # Audit sensitivity: repeat selector fitting with the corrected-unit rule
    # available to both selectors, rather than only compare against it post hoc.
    expanded_summary=[]
    expanded_selected=[]
    for alpha in [.7,.8,.85,.9,.95]:
        for _, row in profiles.iterrows():
            train_rows=val_all[~val_all.store_id.eq(row.store_id)]
            group=train_rows[train_rows.operational_profile.eq(row.operational_profile)]
            held=test_all[np.logical_and.reduce([test_all[k].eq(row[k]) for k in KEYS])].set_index('policy')
            for strategy,rows in [('global',train_rows),('profile',group if len(group) else train_rows)]:
                policy,fallback=choose(rows,alpha)
                expanded_selected.append({k:row[k] for k in KEYS}|{'alpha':alpha,'strategy':strategy,'policy':policy,'fallback':fallback,
                    **{m:float(held.loc[policy,m]) for m in ['TIC','NS','demand','served']}})
    expanded_df=pd.DataFrame(expanded_selected)
    for (alpha,strategy), rows in expanded_df.groupby(['alpha','strategy']):
        expanded_summary.append({'alpha':alpha,'strategy':strategy,'TIC_mean':rows.TIC.mean(),'NS_mean':rows.NS.mean(),
            'demand_weighted_NS':rows.served.sum()/rows.demand.sum(),
            'series_below_alpha':int((rows.NS<alpha).sum()),'fallback_count':int(rows.fallback.sum()),
            'policy_counts':rows.policy.value_counts().to_dict()})
    expanded_df.to_csv(OUT/'expanded_selections.csv',index=False)
    pd.DataFrame(expanded_summary).to_csv(OUT/'expanded_summary.csv',index=False)
    payload={'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'n_series':len(profiles),'n_stores':profiles.store_id.nunique(),
             'profile_counts':profiles.operational_profile.value_counts().to_dict(),
             'calibration_cycles':17,'validation_cycles':10,'test_cycles':11,
             'portfolio':list(POLICIES), 'bootstrap_draws':5000,'bootstrap_unit':'store',
             'seed':20261014, 'device':'cpu; scalar deterministic heuristics',
             'zero_demand_validation_series':int(val.groupby(KEYS).demand.first().eq(0).sum()),
             'zero_demand_test_series':int(test.groupby(KEYS).demand.first().eq(0).sum()),
             'zero_demand_convention':'NS=1, retain cost; positive-demand-only macro NS also reported',
             'audit_amendment':'Zero-demand convention and EOQPerCycle sensitivity added after reviewer audit; original five-policy selector unchanged.',
             'summary':summary,'contrasts':contrasts,'expanded_six_policy_sensitivity':expanded_summary,
             'limitations':['Short reset episodes; no terminal inventory credit.',
                            'Not a rerun or validation of historical RL/GA results.',
                            'Bootstrap conditional on selected strategies; store clusters share products.',
                            'Series inclusion inherited from retrospective prepared cohort.',
                            'Simulator L=2 delivers order t at t+3.']}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
