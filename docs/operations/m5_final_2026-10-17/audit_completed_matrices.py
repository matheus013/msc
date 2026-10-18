from pathlib import Path
import hashlib,json,sqlite3
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
BASE=ROOT/'experiment/data/08_reporting/research_v2'
CASE=['origin','warehouse','store_id','item_id']
report={'scope':'Read-only controller matrix audit; no new controller fitting, no selectors fitted by this script','datasets':{}}
for ds,name in [('BA','aipe_ba145_unified_2026-10-15'),('M5','aipe_m5_2408_unified_2026-10-15')]:
 r=BASE/name
 p=json.loads((r/'plan.json').read_text())
 k=pd.read_csv(r/'kpis.csv',dtype={c:str for c in CASE[1:]},low_memory=False)
 expected={(policy,seed,block) for policy in p['policies'] for seed in (p['seeds'] if policy in {'GA','PPO','GA-DQN','DQN'} else [0]) for block in ['validation','test']}
 assert not k.duplicated(CASE+['policy','seed','block']).any()
 for _,g in k.groupby(CASE):assert set(g[['policy','seed','block']].itertuples(index=False,name=None))==expected
 assert len(k)==2*p['planned_fits']
 assert np.isfinite(k.TIC).all() and k.TIC.ge(0).all() and k.NS.dropna().between(0,1).all()
 case=k.groupby(CASE+['block'],as_index=False).agg(defined=('NS','count'),rows=('NS','size'))
 assert case.apply(lambda x:x.defined in [0,x.rows],axis=1).all()
 test=case[case.block.eq('test')]
 rawmeans=k.groupby(CASE+['policy','block'],as_index=False)[['TIC','NS']].mean()
 rawmeans[rawmeans.block.eq('test')].groupby('policy')[['TIC','NS']].mean().to_csv(OUT/f'{ds}_policy_test_means.csv')
 entry={'run':name,'state':json.loads((r/'status.json').read_text())['state'],'planned_fits':p['planned_fits'],'kpi_rows':len(k),'complete_cases':len(test),'unique_series':len(k[CASE[1:]].drop_duplicates()),'stores':k.store_id.nunique(),'cohorts':p['cohorts'],'undefined_service_test_cases':int(test.defined.eq(0).sum()),'defined_service_test_cases':int(test.defined.gt(0).sum()),'zero_service_by_origin':test.groupby('origin').defined.apply(lambda x:int(x.eq(0).sum())).to_dict(),'kpis_sha256':hashlib.sha256((r/'kpis.csv').read_bytes()).hexdigest()}
 if (r/'selection/summary.csv').exists():
  s=pd.read_csv(r/'selection/summary.csv');s.to_csv(OUT/f'{ds}_selector_summary.csv',index=False)
  assert s.cases.eq(len(test)).all() and s.defined_service.eq(entry['defined_service_test_cases']).all()
  entry['selector_summary_available']=True
 else:entry['selector_summary_available']=False
 report['datasets'][ds]=entry
(OUT/'matrix_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
