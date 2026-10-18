"""Exact bookkeeping checks; estimators are stubbed, never fitted."""
from pathlib import Path
import sys,json,hashlib
from unittest.mock import patch
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'));sys.path.insert(0,str(Path(__file__).parent))
import final_selector_fast as fast
from simulation.research import pse,pse_consequences as original

def run():
 checks=[]
 rows=[]
 for i,vals in enumerate([[(4,.8),(4,.9),(6,.95)],[(4,.6),(5,.6),(3,.59)],[(4,np.nan),(2,np.nan),(2,np.nan)],[(1,.7-1e-9),(1,.7-2e-9),(2,.9)],[(0,1),(0,1),(1,1)]]):
  for policy,(cost,service) in zip(['A','B','C'],vals):rows.append(dict(origin=i+1,warehouse='CA',store_id='same',item_id=str(i),policy=policy,TIC=cost,NS=service))
 frame=pd.DataFrame(rows);frame=pd.concat([frame,frame.assign(TIC=frame.TIC+2)],ignore_index=True)
 for margin in [0.,.02,.1,np.inf,np.nan,-.1,1.]:
  for guarded in [False,True]:pd.testing.assert_frame_equal(original.decisions(frame,'B',margin,guarded),fast.decisions_fast(frame,'B',margin,guarded),check_exact=True)
 for _,g in frame.groupby('origin'):
  a=original.tune_margin([(g,'B')],g);b=fast.tune_margin_fast([(g,'B')],g);assert a[0]==b[0];pd.testing.assert_frame_equal(pd.DataFrame(a[1]),pd.DataFrame(b[1]),check_exact=True)
 checks.append('ties, all-NaN service, tolerance boundary, duplicate policy rows/first raw row, inf/NaN/negative/unit margins')
 datasets=[]
 for ds,name in [('BA','aipe_ba145_unified_2026-10-15'),('M5','aipe_m5_2408_unified_2026-10-15')]:
  r=ROOT/'experiment/data/08_reporting/research_v2'/name
  raw=pd.read_csv(r/'kpis.csv',dtype={k:str for k in pse.CASE[1:]},low_memory=False).query('block=="validation"')
  o=raw.origin.max();raw=raw[raw.origin.eq(o)]
  stores=sorted(raw.store_id.unique())[:4];raw=raw[raw.store_id.isin(stores)]
  features=raw[pse.CASE+['profile']].drop_duplicates().groupby('store_id',group_keys=False).head(10).rename(columns={'profile':'operational_profile'})
  validation=features[pse.CASE].merge(raw,on=pse.CASE,validate='one_to_many')
  calls=[]
  def stub(train,held,with_profile=True):
   calls.append((train.copy(),held.copy(),with_profile))
   labels=sorted(train.label.unique());return {'policy':np.full(len(held),labels[0] if with_profile else labels[-1],dtype=object)}
  with patch.object(pse,'fit_predict_fold',stub),patch.object(fast,'fit_predict_fold',stub):
   a=pse.predict_loso(features,validation);n=len(calls);b=fast.predict_loso_fast(features,validation)
  pd.testing.assert_frame_equal(a,b,check_exact=True);assert len(calls)==2*n
  for x,y in zip(calls[:n],calls[n:]):
   pd.testing.assert_frame_equal(x[0],y[0],check_exact=True);pd.testing.assert_frame_equal(x[1],y[1],check_exact=True);assert x[2]==y[2]
  # Classifier plumbing: inspect exact training rows, labels and held rows;
  # both construct identical fixed estimators but fitting is intercepted.
  train=features[features.store_id.ne(stores[0])];held=features[features.store_id.eq(stores[0])];valid=validation[validation.store_id.ne(stores[0])];observed=[]
  class Model:
   def fit(self,x,y):observed.append((x.copy(),y.copy()));self.label=sorted(set(y))[0];return self
   def predict(self,x):observed.append(x.copy());return np.full(len(x),self.label,dtype=object)
  def pipeline(*models):
   observed.append(models[-1].get_params());return Model()
  with patch.object(original,'make_pipeline',pipeline),patch('sklearn.pipeline.make_pipeline',pipeline):
   a=original.matched_classifier(train,held,valid);cut=len(observed);b=fast.matched_classifier_fast(train,held,valid)
  pd.testing.assert_frame_equal(a,b,check_exact=True)
  assert observed[0]==observed[cut]
  pd.testing.assert_frame_equal(observed[1][0],observed[cut+1][0],check_exact=True);np.testing.assert_array_equal(observed[1][1],observed[cut+1][1]);pd.testing.assert_frame_equal(observed[2],observed[cut+2],check_exact=True)
  datasets.append({'dataset':ds,'origin':int(o),'cases':len(features),'stores':len(stores),'exact_output_rows':len(a),'fit_calls':0,'train_held_order_identical':True,'estimator_parameters_identical':True})
 checks.append('Real BA/M5 class/PSE bookkeeping, exact labels, fixed estimator parameters, training/held row order; no estimator fitting')
 out=ROOT/'docs/operations/m5_final_2026-10-17/fast_semantic_checks.json'
 out.write_text(json.dumps({'status':'all_exact','helper_sha256':hashlib.sha256(Path(fast.__file__).read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks,'real_subsets':datasets,'controller_fits':0,'selector_fits':0},indent=2),encoding='utf8')
 print(out.read_text())
if __name__=='__main__':run()
