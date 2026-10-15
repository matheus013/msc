"""Audit preserved real-retail and M5 matrices with a shared selector rule.

This is an artifact reanalysis, not independent controller retraining.
Store-out selectors are fitted once per excluded store, never per repeat.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sca_validation import choose

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiment/data/08_reporting/sca_article/dual'
KEYS=['warehouse','store_id','item_id']


def audit(name,folder):
    paths={'kpis':ROOT/f'experiment/data/07_model_output/{folder}kpis.parquet',
           'profiles':ROOT/f'experiment/data/04_feature/{folder}demand_profiles.parquet',
           'scenarios':ROOT/f'experiment/data/03_primary/{folder}scenarios.parquet'}
    k,p,s=[pd.read_parquet(paths[n]) for n in ['kpis','profiles','scenarios']]
    for frame in [k,p,s]:
        for key in KEYS:frame[key]=frame[key].astype(str)
    assert not k.duplicated(KEYS+['policy']).any()
    assert not p.duplicated(KEYS).any()
    assert k.groupby(KEYS).policy.nunique().eq(k.policy.nunique()).all()
    df=k.merge(p[KEYS+['operational_profile']],on=KEYS,validate='many_to_one')
    assert len(df)==len(k)
    summaries,choices,contrasts=[],[],[]
    for alpha in [.7,.8,.85,.9,.95]:
        selections=[]
        gp,gf=choose(df,alpha)
        profile_choices={g:choose(rows,alpha) for g,rows in df.groupby('operational_profile')}
        for store,held in df.groupby('store_id'):
            train=df[~df.store_id.eq(store)]
            sg,sfg=choose(train,alpha)
            sp={g:choose(rows,alpha) for g,rows in train.groupby('operational_profile')}
            for (wh,st,it),rows in held.groupby(KEYS):
                profile=rows.operational_profile.iloc[0]
                targets={'global_in_sample':(gp,gf),'profile_in_sample':profile_choices[profile],
                         'global_store_out':(sg,sfg),'profile_store_out':sp.get(profile,(sg,sfg)),
                         'legacy_EOQ':('EOQ',False)}
                for strategy,(policy,fallback) in targets.items():
                    row=rows[rows.policy.eq(policy)].iloc[0]
                    selections.append(dict(zip(KEYS,[wh,st,it]))|{'dataset':name,'alpha':alpha,
                        'profile':profile,'strategy':strategy,'policy':policy,'fallback':bool(fallback),
                        **{metric:float(row[metric]) for metric in ['TIC','NS','TR','BE','FP']}})
        selected=pd.DataFrame(selections)
        selected.to_csv(OUT/f'{name}_selections_{alpha:.2f}.csv',index=False)
        for strategy,rows in selected.groupby('strategy'):
            summaries.append({'dataset':name,'alpha':alpha,'strategy':strategy,
                **{metric+'_mean':float(rows[metric].mean()) for metric in ['TIC','NS','TR','BE','FP']},
                'below_alpha':int((rows.NS<alpha).sum()),'fallback_count':int(rows.fallback.sum()),
                'policy_counts':rows.policy.value_counts().to_dict()})
        a=selected[selected.strategy.eq('global_store_out')].set_index(KEYS).sort_index()
        b=selected[selected.strategy.eq('profile_store_out')].set_index(KEYS).sort_index()
        assert a.index.equals(b.index)
        stores=np.asarray(a.index.get_level_values('store_id').unique())
        group={store:np.flatnonzero(a.index.get_level_values('store_id')==store) for store in stores}
        av,bv=a.TIC.to_numpy(),b.TIC.to_numpy(); svc=b.NS.to_numpy()-a.NS.to_numpy()
        rng=np.random.default_rng(20261014); boot=[]
        for _ in range(5000):
            ix=np.concatenate([group[store] for store in rng.choice(stores,len(stores),replace=True)])
            boot.append([100*(av[ix].sum()-bv[ix].sum())/av[ix].sum(),100*svc[ix].mean()])
        ci=np.quantile(boot,[.025,.975],axis=0)
        contrasts.append({'dataset':name,'alpha':alpha,'cost_reduction_percent':100*(av.sum()-bv.sum())/av.sum(),
            'cost_reduction_ci':ci[:,0].tolist(),'fill_difference_pp':100*svc.mean(),
            'fill_difference_ci_pp':ci[:,1].tolist(),'cluster_stores':len(stores),
            'conditional_bootstrap_draws':5000})
        choices.append({'alpha':alpha,'global':gp,'global_fallback':bool(gf),
                        'profiles':{g:{'policy':v[0],'fallback':bool(v[1])} for g,v in profile_choices.items()}})
    policies=k.groupby('policy')[['TIC','NS','TR','BE','FP']].mean()
    policies['TIC_relative_legacy_EOQ']=policies.TIC/policies.loc['EOQ','TIC']
    policies.to_csv(OUT/f'{name}_policies.csv')
    lengths=s.groupby(KEYS).size()
    prefix=17 if name=='real_BA' else 19
    zero_train=zero_eval=few_train=0
    for _,rows in s.sort_values('venda_ciclo').groupby(KEYS):
        demand=rows.demand.to_numpy(float)
        zero_train+=int(demand[:prefix].sum()==0)
        zero_eval+=int(demand[prefix:].sum()==0)
        few_train+=int((demand[:prefix]>0).sum()<3)
    payload={'dataset':name,'n_series':len(p),'n_stores':s.store_id.nunique(),'n_items':s.item_id.nunique(),
       'warehouses':sorted(s.warehouse.unique()),'n_policies':k.policy.nunique(),'rows':len(k),
       'cycles_min':int(lengths.min()),'cycles_max':int(lengths.max()),
       'configured_prefix_cycles':prefix,'zero_demand_prefix_series':zero_train,
       'fewer_than_three_positive_prefix_cycles':few_train,'zero_demand_eval_series':zero_eval,
       'profile_counts':p.operational_profile.value_counts().to_dict(),
       'group_counts':k[KEYS+['group']].drop_duplicates()['group'].value_counts().to_dict(),
       'maximum_TIC_std':float(k.TIC_std.max()),
       'missing_by_column':{c:int(v) for c,v in k.isna().sum().items() if v},
       'input_sha256':{n:hashlib.sha256(path.read_bytes()).hexdigest() for n,path in paths.items()},
       'summaries':summaries,'selection_rules':choices,'contrasts':contrasts}
    (OUT/f'{name}_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    payload={'date':'2026-10-14','cohorts':[audit('real_BA',''),audit('M5','m5/')],
             'qualification':'Conditional reanalysis of saved matrices; profiles/preprocessing may use full series; not PSE generalization or retraining.'}
    (OUT/'dual_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    pd.DataFrame([r for p in payload['cohorts'] for r in p['summaries']]).to_csv(OUT/'strategy_summary.csv',index=False)
    for p in payload['cohorts']:
        print(p['dataset'],p['n_series'],p['n_stores'],p['contrasts'][0])
        print(pd.DataFrame(p['summaries']).query('alpha==0.7').to_string(index=False))


if __name__=='__main__':main()
