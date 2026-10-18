"""Equivalent vectorized selector bookkeeping outside frozen simulation modules.

Not active in canonical execution. Does not fit inventory controllers. Benchmark
mode uses cached training-prefix features/validation outcomes only.
"""
from pathlib import Path
import os,sys,time,json,argparse,hashlib
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiment/src'))
import numpy as np,pandas as pd
from pandas.testing import assert_frame_equal
from simulation.research.pse import CASE,best_policy,fit_predict_fold,feature_transform
from simulation.research import pse_consequences as original

def best_labels(validation,keys=CASE,alpha=.7):
 mean=validation.groupby([*keys,'policy'],as_index=False,sort=True)[['TIC','NS']].mean()
 feasible=mean[mean.NS.ge(alpha-1e-9)].sort_values([*keys,'TIC','policy']).drop_duplicates(keys)
 fallback=mean.sort_values([*keys,'NS','TIC','policy'],ascending=[True]*len(keys)+[False,True,True],na_position='last').drop_duplicates(keys)
 index=pd.MultiIndex.from_frame(feasible[keys])
 rest=fallback[~pd.MultiIndex.from_frame(fallback[keys]).isin(index)]
 winners=pd.concat([feasible,rest],ignore_index=True).sort_values(keys)
 return winners[[*keys,'policy']].set_index(keys).policy

def decisions_fast(predicted,reference,margin=0.,guarded=False):
 if predicted.empty:return pd.DataFrame()
 names=['policy','reference_policy','candidate_policy','intervened','margin','predicted_cost','predicted_service','reference_predicted_cost','reference_predicted_service']
 labels=best_labels(predicted).rename('candidate_policy').reset_index()
 raw=predicted.drop_duplicates(CASE+['policy'])
 selected=labels.merge(raw.rename(columns={'policy':'candidate_policy','TIC':'predicted_cost','NS':'predicted_service'}),on=CASE+['candidate_policy'],validate='one_to_one')
 baseline=raw[raw.policy.eq(reference)][CASE+['TIC','NS']].rename(columns={'TIC':'reference_predicted_cost','NS':'reference_predicted_service'})
 if len(baseline)!=len(labels):raise IndexError('single positional indexer is out-of-bounds')
 rows=selected.merge(baseline,on=CASE,validate='one_to_one').sort_values(CASE).reset_index(drop=True)
 intervene=(np.isfinite(margin)&rows.predicted_service.ge(.7-1e-9)&rows.predicted_service.ge(rows.reference_predicted_service-1e-9)&rows.predicted_cost.lt(rows.reference_predicted_cost*(1-margin)-1e-9))
 rows['policy']=np.where(intervene,rows.candidate_policy,reference) if guarded else rows.candidate_policy
 rows['reference_policy']=reference;rows['intervened']=rows.policy.ne(reference);rows['margin']=margin
 return rows[CASE+names]

def tune_margin_fast(inner_predictions,validation):
 truth=validation.groupby(CASE+['policy'],as_index=False)[['TIC','NS']].mean()
 choices=pd.concat([decisions_fast(p,ref) for p,ref in inner_predictions],ignore_index=True)
 baselines=choices[CASE+['reference_policy']].rename(columns={'reference_policy':'policy'}).merge(truth,on=CASE+['policy'],validate='one_to_one')
 candidate=choices[CASE+['candidate_policy']].rename(columns={'candidate_policy':'policy'}).merge(truth,on=CASE+['policy'],validate='one_to_one')
 # Candidate and baseline lookup keep left CASE order exactly as original choices.
 rows=[]
 for margin in original.MARGINS:
  intervene=(np.isfinite(margin)&choices.predicted_service.ge(.7-1e-9)&choices.predicted_service.ge(choices.reference_predicted_service-1e-9)&choices.predicted_cost.lt(choices.reference_predicted_cost*(1-margin)-1e-9))
  tic=np.where(intervene,candidate.TIC,baselines.TIC);ns=np.where(intervene,candidate.NS,baselines.NS)
  policies=np.where(intervene,choices.candidate_policy,choices.reference_policy)
  rows.append({'margin':margin,'cost':float(pd.Series(tic).mean()),'service':float(pd.Series(ns).mean()),'reference_service':float(baselines.NS.mean()),'interventions':int((policies!=choices.reference_policy).sum())})
 candidates=[r for r in rows if r['service']>=r['reference_service']-1e-9]
 if not candidates:return float('inf'),rows
 return min(candidates,key=lambda r:(r['cost'],-r['service'],-r['margin']))['margin'],rows

def matched_classifier_fast(train,held,validation):
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.pipeline import make_pipeline
 labels=best_labels(validation)
 target=labels.reindex(pd.MultiIndex.from_frame(train[CASE])).to_numpy()
 model=make_pipeline(feature_transform(),ExtraTreesClassifier(n_estimators=64,max_depth=4,min_samples_leaf=3,random_state=20261015,n_jobs=1))
 model.fit(train,target)
 rows=held[CASE].copy();rows['policy']=model.predict(held);rows['strategy']='Classe_floresta_controle'
 return rows

def predict_loso_fast(features,validation,alpha=.7):
 rows=[];labels=best_labels(validation,alpha=alpha)
 data=features.set_index(CASE).join(labels.rename('label')).reset_index()
 if data.label.isna().any():raise ValueError('Missing validation label')
 for origin,cohort in data.groupby('origin'):
  if cohort.store_id.nunique()<2:raise ValueError('At least two stores required')
  for store in sorted(cohort.store_id.unique()):
   train=cohort[cohort.store_id.ne(store)];held=cohort[cohort.store_id.eq(store)]
   valid_train=validation[validation.origin.eq(origin)&validation.store_id.ne(store)]
   global_policy=best_policy(valid_train,alpha)
   mode=train.label.value_counts().sort_index().sort_values(ascending=False,kind='stable').index[0]
   chosen={}
   for strategy,profile in [('AIPE_PSE',True),('PSE_sem_perfil',False)]:chosen[strategy]=fit_predict_fold(train,held,profile)['policy']
   profile_train=valid_train.merge(train[CASE+['operational_profile']],on=CASE)
   profiles=best_labels(profile_train,keys=['operational_profile'],alpha=alpha)
   seen=set(train.label)
   for pos,(_,case) in enumerate(held.iterrows()):
    selections={s:v[pos] for s,v in chosen.items()};selections.update(Global=global_policy,Perfil=profiles.get(case.operational_profile,global_policy),Modo_rotulos=mode)
    for strategy,policy in selections.items():rows.append({k:case[k] for k in CASE}|{'strategy':strategy,'policy':policy,'validation_label':case.label,'label_unseen_in_training':case.label not in seen})
 return pd.DataFrame(rows)

def benchmark():
 from simulation.research.pse import predict_loso
 records=[]
 rng=np.random.default_rng(11)
 for ds,name in [('BA','aipe_ba145_unified_2026-10-15'),('M5','aipe_m5_2408_unified_2026-10-15')]:
  run=ROOT/'experiment/data/08_reporting/research_v2'/name
  validation=pd.read_csv(run/'kpis.csv',dtype={k:str for k in CASE[1:]},low_memory=False).query('block=="validation"')
  # Actual validation outcomes form complete real-valued predicted matrices for
  # validating decision bookkeeping; this is not an out-of-sample experiment.
  matrix=validation.groupby(CASE+['policy'],as_index=False)[['TIC','NS']].mean()
  for origin,m in matrix.groupby('origin'):
   reference=best_policy(m)
   t=time.perf_counter();labels=validation[validation.origin.eq(origin)].groupby(CASE).apply(best_policy,include_groups=False);slow=time.perf_counter()-t
   t=time.perf_counter();fast=best_labels(validation[validation.origin.eq(origin)]);quick=time.perf_counter()-t
   pd.testing.assert_series_equal(labels,fast,check_names=False)
   records.append({'dataset':ds,'origin':int(origin),'operation':'labels','slow_seconds':slow,'fast_seconds':quick})
   for guarded in [False,True]:
    for margin in [0.,.02,.05,.1,np.inf]:
     t=time.perf_counter();a=original.decisions(m,reference,margin,guarded);slow=time.perf_counter()-t
     t=time.perf_counter();b=decisions_fast(m,reference,margin,guarded);quick=time.perf_counter()-t
     assert_frame_equal(a,b,check_exact=True)
     records.append({'dataset':ds,'origin':int(origin),'operation':f'decisions-{guarded}-{margin}','slow_seconds':slow,'fast_seconds':quick})
   inner=[(part,reference) for _,part in m.groupby('store_id')]
   t=time.perf_counter();a=original.tune_margin(inner,validation[validation.origin.eq(origin)]);slow=time.perf_counter()-t
   t=time.perf_counter();b=tune_margin_fast(inner,validation[validation.origin.eq(origin)]);quick=time.perf_counter()-t
   assert a[0]==b[0];assert_frame_equal(pd.DataFrame(a[1]),pd.DataFrame(b[1]),check_exact=True)
   records.append({'dataset':ds,'origin':int(origin),'operation':'tune_margin','slow_seconds':slow,'fast_seconds':quick})
 out=ROOT/'docs/operations/m5_final_2026-10-17/fast_equivalence.json'
 out.write_text(json.dumps({'status':'all_exact','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'controller_fits':0,'models_fitted':0,'tests':records},indent=2),encoding='utf8')
 print(json.dumps(records,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['benchmark']);args=parser.parse_args();benchmark()
