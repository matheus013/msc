"""Read-only context audit: aggregate coverage; no policy/model fitting."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import duckdb,pandas as pd,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiment/data/audits/context_variables_2026-10-17';OUT.mkdir(parents=True,exist_ok=True)
run=ROOT/'experiment/data/08_reporting/research_v2/aipe_ba145_unified_2026-10-15'
plan=json.loads((run/'plan.json').read_text(encoding='utf-8'))
panel=pd.read_parquet(run/plan['panel_metadata'][0]['panel'])
keys=panel.index.to_frame(index=False).astype(str)
con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='768MB'")
con.register('cohort',keys)
attrs=['genero','idade','segmento','status','estrutura','filial','praca','gerente_regional','ciclos_inativos']
source=(ROOT/'data/source/vendas/uf=BA/*.parquet').as_posix()
con.execute(f"CREATE TEMP TABLE selected AS SELECT * FROM (SELECT CAST(revendedor_cod AS VARCHAR) AS store_id, CAST(produto_cod AS VARCHAR) AS item_id, venda_ciclo, venda_data, {','.join(attrs)} FROM read_parquet('{source}', hive_partitioning=false)) AS raw SEMI JOIN cohort USING(store_id,item_id)")
def q(sql):return json.loads(con.execute(sql).df().to_json(orient='records',date_format='iso'))
report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'existing context availability, not outcome gains or historical validity','cohort':{'series':len(keys),'stores':keys.store_id.nunique(),'items':keys.item_id.nunique()},'rows':q('SELECT count(*) n, min(venda_ciclo) first_cycle,max(venda_ciclo) last_cycle,count(DISTINCT store_id) stores_matched FROM selected')[0],'fields':{},'source_schemas':{}}
for folder in ['seller','vendas','segmentacao','produtos_ativos']:
 files=list((ROOT/'data/source'/folder).rglob('*.parquet'))
 report['source_schemas'][folder]={'files':len(files),'columns':pq.ParquetFile(files[0]).schema_arrow.names}
for c in attrs:
 clean=f"NULLIF(TRIM(CAST(\"{c}\" AS VARCHAR)),'')"
 stats=q(f'''WITH by_store AS (SELECT store_id,count({clean}) nonnull,count(DISTINCT {clean}) n_values FROM selected GROUP BY store_id)
 SELECT sum(CASE WHEN nonnull>0 THEN 1 ELSE 0 END) stores_with_value,sum(CASE WHEN n_values>1 THEN 1 ELSE 0 END) stores_with_multiple_values FROM by_store''')[0]
 stats.update(q(f'SELECT count(*) observations,count({clean}) observations_with_value,count(DISTINCT {clean}) distinct_values FROM selected')[0])
 if c in ['genero','segmento','status','ciclos_inativos']:
  stats['categories']=q(f'SELECT {clean} category,count(DISTINCT store_id) stores FROM selected GROUP BY category ORDER BY stores DESC LIMIT 20')
 report['fields'][c]=stats
report['age_summary']=q('SELECT min(idade) minimum,max(idade) maximum,median(idade) median,count(*) FILTER(WHERE idade<18 OR idade>100) outside_18_100_rows FROM selected')[0]
report['origin_prefix_coverage']=[]
for origin in [17,23]:
 end=int(str(panel.columns[origin-1]));r=q(f'SELECT count(DISTINCT store_id) stores_with_any_sale,count(*) n_rows FROM selected WHERE venda_ciclo<={end}')[0];r.update(origin=origin,last_training_cycle=end);report['origin_prefix_coverage'].append(r)
# The commercial segmentation is a reference mapping, not a seller-income measure.
seg=(ROOT/'data/source/segmentacao/*.parquet').as_posix()
report['commercial_segmentation_reference']=q(f"SELECT DISTINCT plano,modelo_atendimento,segmentacao,a_partir,ate FROM read_parquet('{seg}') ORDER BY plano,a_partir")
seller=(ROOT/'data/source/seller/uf=BA/*.parquet').as_posix()
report['seller_cohort_join']=q(f"SELECT count(*) matched_rows,count(DISTINCT CAST(id AS VARCHAR)) matched_stores FROM read_parquet('{seller}',hive_partitioning=false) WHERE CAST(id AS VARCHAR) IN (SELECT DISTINCT store_id FROM cohort)")[0]
# Actual canonical selector inputs and saved feature schemas.
features=[]
for p in sorted(run.rglob('*.csv')):
 if 'feature' in p.name.lower():features.append({'file':str(p.relative_to(ROOT)),'columns':pd.read_csv(p,nrows=0).columns.tolist()})

for p in sorted(run.rglob('*.parquet')):
 if 'feature' in p.name.lower():features.append({'file':str(p.relative_to(ROOT)),'columns':pq.ParquetFile(p).schema_arrow.names})
report['canonical_feature_files']=features
m5=ROOT/'data/m5_walmart'
report['m5_calendar_columns']=pd.read_csv(m5/'calendar.csv',nrows=0).columns.tolist()
report['m5_calendar_summary']={c:{'distinct':int(pd.read_csv(m5/'calendar.csv',usecols=[c])[c].nunique()),'missing':int(pd.read_csv(m5/'calendar.csv',usecols=[c])[c].isna().sum())} for c in ['snap_CA','snap_TX','snap_WI','event_name_1','event_name_2']}
(OUT/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
