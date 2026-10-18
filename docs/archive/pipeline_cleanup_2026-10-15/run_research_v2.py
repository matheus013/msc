"""Causal pilot on restored BA sales and unmodified, complete M5 cycles.

python experiment/scripts/analysis/run_research_v2.py --max-series 8
Use --include-rl only for an implementation pilot; neural budgets here are small.
Writes exclusively to experiment/data/08_reporting/research_v2/<run>.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import pandas as pd
import duckdb

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.protocol import calibration_config,fit_policy,simulate,common_initial_state,profile_of,DETERMINISTIC
from simulation.pipelines.data_ingestion.m5_loader import load_m5_as_internal
from simulation.core.device import get_device

KEYS=['warehouse','store_id','item_id']


def load_panel(name):
    if name=='BA':
        source=ROOT/'experiment/data/02_intermediate/sales_raw.parquet'
        con=duckdb.connect()
        panel=con.execute('''SELECT upper(_uf) warehouse,CAST(revendedor_cod AS VARCHAR) store_id,
            CAST(produto_cod AS VARCHAR) item_id,CAST(venda_ciclo AS VARCHAR) venda_ciclo,
            sum(TRY_CAST(venda_qtd AS DOUBLE)) demand FROM read_parquet(?)
            WHERE upper(_uf)='BA' AND TRY_CAST(venda_qtd AS DOUBLE)>=0
            GROUP BY ALL''',[str(source)]).df();con.close()
        note='Pre-filter transaction snapshot; sum all rows by cycle, no key-first dedup, no full-history CV/activity filter. Not yet independently reconciled with original transaction IDs.'
    else:
        source=ROOT/'data/m5_walmart/sales_train_evaluation.csv'
        panel=load_m5_as_internal({'cycles_per_year':17,'m5':{'path':str(source.parent),
              'categories':['FOODS'],'max_cycles':92,'max_series':None,'iqr_outlier_smoothing':False,'with_revenue':False}})
        note='FOODS raw quantities; 92 full 21-day cycles; no IQR clipping or future CV filter.'
    panel=panel[KEYS+['venda_ciclo','demand']]
    for k in KEYS+['venda_ciclo']:panel[k]=panel[k].astype(str)
    assert not panel.duplicated(KEYS+['venda_ciclo']).any()
    wide=panel.pivot(index=KEYS,columns='venda_ciclo',values='demand').sort_index(axis=1).fillna(0)
    return wide,{'source':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'preparation':note,'available_series':len(wide),'complete_cycles':wide.shape[1]}


def select_policy(rows,alpha):
    means=rows.groupby('policy')[['TIC','NS']].mean()
    if means.empty or means.NS.isna().all():
        return None,True  # no demand: service cannot identify a feasible rule
    # Service undefined at zero demand is not silently filled as infeasible.
    valid=means[means.NS>=alpha].sort_values(['TIC','NS'],ascending=[True,False])
    if len(valid):return valid.index[0],False
    return means.sort_values(['NS','TIC'],ascending=[False,True],na_position='last').index[0],True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-series',type=int,default=8)
    parser.add_argument('--include-rl',action='store_true');parser.add_argument('--run',default='pilot_2026-10-14')
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run name')
    output=ROOT/'experiment/data/08_reporting/research_v2'/args.run
    output.mkdir(parents=True,exist_ok=False)
    rows,cohorts,provenance=[],[],{}
    policies=list(DETERMINISTIC)+['GA']+(['DQN','PPO'] if args.include_rl else [])
    seeds=[11,29,47]
    start=time.perf_counter()
    for dataset,origins in [('BA',[17,23]),('M5',[50,64,78])]:
        wide,meta=load_panel(dataset);provenance[dataset]=meta
        for origin in origins:
            assert origin+14<=wide.shape[1]
            past=wide.iloc[:,:origin];npositive=(past>0).sum(axis=1)
            eligible=wide[npositive>=3]
            # Sample only by stable IDs and past eligibility; not cost/test service.
            keys=sorted(eligible.index,key=lambda k:hashlib.sha256('|'.join(k).encode()).hexdigest())
            keys=keys[:args.max_series] if args.max_series else keys
            cohorts.append({'dataset':dataset,'origin':origin,'candidate_series':len(wide),
                'cold_start_zero_prefix':int((npositive==0).sum()),'insufficient_positive_history':int((npositive<3).sum()),
                'eligible_series':len(eligible),'evaluated_series':len(keys),'validation_cycles':7,'test_cycles':7})
            print(dataset,'origin',origin,'eligible',len(eligible),'pilot',len(keys),flush=True)
            for key in keys:
                demand=wide.loc[key].to_numpy(float);train=demand[:origin];validation=demand[origin:origin+7];test=demand[origin+7:origin+14]
                profile=profile_of(train);config=calibration_config(train)
                valid_cfg,test_cfg=common_initial_state(train,validation,config)
                for policy in policies:
                    for seed in (seeds if policy not in DETERMINISTIC else [0]):
                        config=calibration_config(train,seed);beg=time.perf_counter()
                        fitted=fit_policy(policy,train,config);fit_seconds=time.perf_counter()-beg
                        for block,values,episode_cfg in [('validation',validation,valid_cfg),('test',test,test_cfg)]:
                            metrics,_=simulate(values,episode_cfg,fitted)
                            rows.append(dict(zip(KEYS,key))|{'dataset':dataset,'origin':origin,'profile':profile,
                                'policy':policy,'seed':seed,'block':block,'fit_seconds':fit_seconds,
                                'TIC':metrics['TIC'],'NS':metrics['ServiceLevel'],'TR':metrics['StockoutRate'],
                                'BE':metrics['BullwhipEffect'],'FP':metrics['OrderFrequency'],
                                'Demand':metrics['DemandTotal'],'Served':metrics['DemandServed'],
                                'TerminalCommittedUnits':metrics['TerminalCommittedUnits']})
            pd.DataFrame(rows).to_csv(output/'kpis.csv',index=False)
    results=pd.DataFrame(rows);selected=[]
    # Seed is independent training uncertainty, not a new independent retail unit.
    averaged=results.groupby(['dataset','origin']+KEYS+['profile','policy','block'])[['TIC','NS','Demand','Served']].mean().reset_index()
    for (dataset,origin),group in averaged.groupby(['dataset','origin']):
        val=group[group.block.eq('validation')];test=group[group.block.eq('test')]
        for store,held in test.groupby('store_id'):
            other=val[~val.store_id.eq(store)]
            if other.empty:continue
            for alpha in [.7,.9]:
                global_policy,global_fallback=select_policy(other,alpha)
                if global_policy is None:continue
                for key,g in held.groupby(KEYS):
                    profile=g.profile.iloc[0];matching=other[other.profile.eq(profile)]
                    profile_policy,profile_fallback=select_policy(matching if len(matching) else other,alpha)
                    if profile_policy is None:
                        profile_policy=global_policy
                        profile_fallback=True
                    for strategy,policy,fallback in [('global',global_policy,global_fallback),('profile',profile_policy,profile_fallback)]:
                        record=g[g.policy.eq(policy)].iloc[0]
                        selected.append(dict(zip(KEYS,key))|{'dataset':dataset,'origin':origin,'alpha':alpha,
                           'strategy':strategy,'policy':policy,'fallback':fallback,
                           'TIC':record.TIC,'NS':record.NS,'Demand':record.Demand,'Served':record.Served})
    selections=pd.DataFrame(selected);selections.to_csv(output/'selections.csv',index=False)
    summary=[]
    for (dataset,origin,alpha,strategy),g in selections.groupby(['dataset','origin','alpha','strategy']):
        summary.append({'dataset':dataset,'origin':int(origin),'alpha':alpha,'strategy':strategy,
            'series':len(g),'TIC_per_cycle':g.TIC.mean()/7,'mean_NS_positive_demand':g.NS.mean(),
            'volume_weighted_NS':g.Served.sum()/g.Demand.sum() if g.Demand.sum() else None,
            'below_alpha':int((g.NS<alpha).sum()),'undefined_NS':int(g.NS.isna().sum()),
            'fallbacks':int(g.fallback.sum()),'policy_counts':g.policy.value_counts().to_dict()})
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    manifest={'status':'implementation pilot, not publication evidence','device':str(get_device()),'policies':policies,
       'training_seeds':seeds,'sources':provenance,'cohorts':cohorts,'elapsed_seconds':time.perf_counter()-start,
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'config_by_dataset':'same calibration_config per method; GA pilot 32 individuals/8 generations, optional RL 10 episodes; not equal total compute across algorithm families',
       'core_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (ROOT/'experiment/src/simulation').rglob('*.py')},
       'protocol':'Past-only preparation/fit/profiles. Independent retraining per stochastic seed. Test inventory/pipeline common to candidates, from fixed EOQ baseline history. Seven-cycle validation/test; store-out selection.',
       'limitations':['Small hashed pilot sample; no population inference.','BA source is preserved raw transaction snapshot, not independently reconciled IDs.','No terminal salvage credit; terminal committed units exported for sensitivity.','No neural methods unless --include-rl; their pilot episodes are insufficient for algorithm rankings.','Cold-start histories counted separately, not evaluated as mature series.','No claim of prospective supervised PSE validation.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Pilot written:',output,flush=True)


if __name__=='__main__':main()
