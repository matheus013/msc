"""Read-only input-contract audit; no scientific imports, fitting or simulation.

Run from project root with experiment/.venv/Scripts/python.exe.
Outputs local aggregate evidence, never raw personal records.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sqlite3
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiment/data/audits/input_contract_2026-10-16'
BASE = ROOT / 'experiment/data/08_reporting/research_v2'
RUNS = {'BA': 'aipe_ba145_unified_2026-10-15', 'M5': 'aipe_m5_2408_unified_2026-10-15'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def query(con, sql, args=None):
    frame = con.execute(sql, args or []).df()
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'input availability and consumption, not outcome evaluation',
                'source_inventory': {}, 'datasets': {}, 'checks': []}
    con = duckdb.connect()
    con.execute('SET threads=1')
    con.execute("SET memory_limit='768MB'")
    for folder in sorted((ROOT/'data/source').iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.rglob('*.parquet'))
        schemas = {}
        rows = 0
        for f in files:
            p = pq.ParquetFile(f)
            schema = {x.name: str(x.type) for x in p.schema_arrow}
            sig = json.dumps(schema, sort_keys=True)
            schemas.setdefault(sig, {'fields': schema, 'files': 0})['files'] += 1
            rows += p.metadata.num_rows
        evidence['source_inventory'][folder.name] = {
            'files': len(files), 'rows_from_metadata': rows,
            'bytes': sum(f.stat().st_size for f in files), 'schemas': list(schemas.values())}

    for ds, name in RUNS.items():
        run = BASE/name
        plan = json.loads((run/'plan.json').read_text(encoding='utf-8'))
        d = {'run': name, 'plan_sha256': sha(run/'plan.json'),
             'status_at_read': json.loads((run/'status.json').read_text()), 'panels': []}
        for meta in plan['panel_metadata']:
            path = run/meta['panel']
            panel = pd.read_parquet(path)
            a = panel.to_numpy(float)
            o = meta['origin']
            digest = sha(path)
            checks = {'hash_matches_manifest': digest == meta['panel_sha256'],
                      'unique_series': not panel.index.duplicated().any(),
                      'finite_nonnegative': bool(np.isfinite(a).all() and (a >= 0).all()),
                      'minimum_three_positive_training_cycles': bool(((a[:, :o] > 0).sum(axis=1) >= 3).all()),
                      'validation_and_test_available': a.shape[1] >= o + 14}
            evidence['checks'].append({'dataset': ds, 'origin': o, **checks})
            d['panels'].append({'origin': o, 'shape': list(panel.shape),
                'index_names': list(panel.index.names), 'stores': panel.index.get_level_values('store_id').nunique(),
                'items': panel.index.get_level_values('item_id').nunique(),
                'first_period': str(panel.columns[0]), 'last_period': str(panel.columns[-1]),
                'train_last_period': str(panel.columns[o-1]),
                'validation_periods': [str(x) for x in panel.columns[o:o+7]],
                'test_periods': [str(x) for x in panel.columns[o+7:o+14]],
                'zero_cells': int((a == 0).sum()), 'source_metadata': meta, 'sha256': digest})
        last = pd.read_parquet(run/plan['panel_metadata'][-1]['panel'])
        con.register(ds.lower()+'_keys', last.index.to_frame(index=False).astype(str))
        # One existing payload is sufficient to inspect persisted fields; no matrix scan or writes.
        with sqlite3.connect((run/'results.sqlite').as_uri()+'?mode=ro', uri=True, timeout=5) as db:
            row = db.execute('SELECT policy,seed,payload FROM fits LIMIT 1').fetchone()
            payload = json.loads(row[2])
            d['persisted_sample'] = {'policy': row[0], 'seed': row[1],
                'payload_fields': list(payload), 'metric_fields': list(payload['rows'][0]),
                'simulation_config': payload['config']['SIMULATION'], 'cost_config': payload['config']['COST'],
                'note': 'sample only; not an assertion that every payload is identical'}
        # Verify frozen copies used as evidence rather than relying only on current source.
        files = ['experiment/src/simulation/research/protocol.py',
                 'experiment/src/simulation/research/production.py',
                 'experiment/src/simulation/research/parallel.py',
                 'experiment/src/simulation/core/inventory_env.py',
                 'experiment/scripts/analysis/run_production_v2.py',
                 'experiment/src/simulation/pipelines/data_ingestion/m5_loader.py']
        d['frozen_dependencies'] = []
        for rel in files:
            frozen = run/'code_snapshot'/rel
            actual = sha(frozen)
            expected = plan['code_sha256'].get(str(Path(rel)))
            check = {'file': rel, 'frozen_sha256': actual,
                     'matches_manifest': actual == expected, 'current_equals_frozen': sha(ROOT/rel) == actual}
            d['frozen_dependencies'].append(check)
            evidence['checks'].append({'dataset': ds, 'file': rel, 'snapshot_integrity': check['matches_manifest']})
        evidence['datasets'][ds] = d

    raw_ba = sorted((ROOT/'data/source/vendas/uf=BA').glob('*.parquet'))
    # Read only selected sales columns, semijoin historical cohort; no personal values exported.
    con.read_parquet([str(x) for x in raw_ba], hive_partitioning=False).create_view('ba_raw')
    evidence['ba_raw_selected_cohort'] = query(con, '''
        SELECT count(*) AS row_count, count(DISTINCT CAST(venda_ciclo AS VARCHAR)) cycles,
            min(venda_data) first_date_text, max(venda_data) last_date_text,
            count(*) FILTER (WHERE try_cast(venda_qtd AS DOUBLE) IS NULL) invalid_quantity,
            count(*) FILTER (WHERE try_cast(venda_qtd AS DOUBLE)<0) negative_quantity,
            count(*) FILTER (WHERE venda_vlr_receita_liquida IS NOT NULL) rows_with_net_revenue
        FROM ba_raw r SEMI JOIN ba_keys k ON CAST(r.revendedor_cod AS VARCHAR)=k.store_id
            AND CAST(r.produto_cod AS VARCHAR)=k.item_id''')
    stocks = sorted((ROOT/'data/source/estoque').glob('*.parquet'))
    maps = sorted((ROOT/'data/source/estoques_rev').glob('*.parquet'))
    con.read_parquet([str(x) for x in stocks], union_by_name=True).create_view('stocks')
    con.read_parquet([str(x) for x in maps], union_by_name=True).create_view('stock_mapping')
    evidence['stock_available_not_consumed'] = query(con, '''SELECT count(*) AS row_count,
        min(dt_estoque) first_snapshot,max(dt_estoque) last_snapshot,
        count(DISTINCT dt_estoque) snapshot_times,count(DISTINCT cod_franquia) franchises,
        count(DISTINCT cod_material) materials,
        count(*) FILTER (WHERE stk_disponivel IS NULL) missing_available,
        count(*) FILTER (WHERE stk_disponivel<0) negative_available FROM stocks''')
    evidence['ba_stock_join_coverage'] = query(con, '''
        WITH mappings AS (SELECT DISTINCT CAST(revendedor_cod AS VARCHAR) seller,
            CAST(estoque_cod AS VARCHAR) stock FROM stock_mapping),
        stocks_keys AS (SELECT DISTINCT CAST(cod_franquia AS VARCHAR) stock,
            CAST(cod_material AS VARCHAR) item FROM stocks),
        coverage AS (SELECT k.store_id,k.item_id,count(DISTINCT m.stock) mapped_stock_ids,
            count(DISTINCT s.stock) matching_stock_material_ids
            FROM ba_keys k LEFT JOIN mappings m ON k.store_id=m.seller
            LEFT JOIN stocks_keys s ON m.stock=s.stock AND k.item_id=s.item
            GROUP BY k.store_id,k.item_id)
        SELECT count(*) cohort_series,count(*) FILTER (WHERE mapped_stock_ids>0) with_mapping,
            count(*) FILTER (WHERE mapped_stock_ids>1) multiple_mapping,
            count(*) FILTER (WHERE matching_stock_material_ids>0) with_stock_material_match FROM coverage''')
    evidence['ba_raw_stock_direct_key_overlap'] = query(con, '''SELECT count(*) cohort_series,
        count(*) FILTER (WHERE s.stock IS NOT NULL) direct_seller_franchise_and_product_match
        FROM ba_keys k LEFT JOIN (SELECT DISTINCT CAST(cod_franquia AS VARCHAR) stock,
        CAST(cod_material AS VARCHAR) item FROM stocks) s ON k.store_id=s.stock AND k.item_id=s.item''')

    # Reconcile quantities actually consumed by frozen panels with current raw inputs.
    ba_agg = con.execute('''SELECT 'BA' warehouse, CAST(r.revendedor_cod AS VARCHAR) store_id,
        CAST(r.produto_cod AS VARCHAR) item_id, CAST(venda_ciclo AS VARCHAR) period,
        sum(try_cast(venda_qtd AS DOUBLE)) quantity FROM ba_raw r SEMI JOIN ba_keys k
        ON CAST(r.revendedor_cod AS VARCHAR)=k.store_id AND CAST(r.produto_cod AS VARCHAR)=k.item_id
        WHERE try_cast(venda_qtd AS DOUBLE)>=0 AND isfinite(try_cast(venda_qtd AS DOUBLE))
        GROUP BY ALL''').df()
    ba_panel = pd.read_parquet(BASE/RUNS['BA']/evidence['datasets']['BA']['panels'][0]['source_metadata']['panel'])
    raw_wide = ba_agg.pivot(index=['warehouse','store_id','item_id'],columns='period',values='quantity')
    raw_wide = raw_wide.reindex(index=ba_panel.index, columns=ba_panel.columns)
    evidence['ba_panel_reconciliation'] = {'unrecorded_cells_filled_zero':int(raw_wide.isna().sum().sum()),
        'cells':int(ba_panel.size),'max_absolute_difference_after_zero_fill':float(np.abs(raw_wide.fillna(0).to_numpy()-ba_panel.to_numpy()).max())}
    evidence['checks'].append({'dataset':'BA','raw_quantities_match_frozen_panel':bool(np.array_equal(raw_wide.fillna(0).to_numpy(),ba_panel.to_numpy()))})

    m5root = ROOT/'data/m5_walmart'
    evidence['m5_files'] = []
    for f in sorted(m5root.glob('*.csv')):
        with f.open(encoding='utf-8-sig',newline='') as stream:
            header = next(csv.reader(stream))
        evidence['m5_files'].append({'name': f.name, 'bytes': f.stat().st_size,
            'columns_count': len(header), 'non_day_fields': [x for x in header if not x.startswith('d_')],
            'day_column_count': sum(x.startswith('d_') for x in header)})
    cal = pd.read_csv(m5root/'calendar.csv',usecols=['date'])
    cal_dates = pd.to_datetime(cal.date)
    calendar_consecutive = bool(cal_dates.diff().dropna().eq(pd.Timedelta(days=1)).all())
    evidence['checks'].append({'file': 'calendar.csv', 'consecutive_daily_rows': calendar_consecutive})
    evidence['m5_real_calendar'] = {'aggregated_days': 92*21, 'discarded_sales_days': 1941-92*21,
        'first_date': cal.iloc[0]['date'], 'last_used_date': cal.iloc[92*21-1]['date'],
        'mapping_note': 'M5 YYYYCC labels are synthetic; this local calendar has no d column; map d_1 to its first chronological daily row.'}
    for d in evidence['datasets']['M5']['panels']:
        o = d['origin']
        d['real_date_ranges'] = {name: [cal.iloc[a]['date'],cal.iloc[b-1]['date']]
            for name,a,b in [('train',0,o*21),('validation',o*21,(o+7)*21),('test',(o+7)*21,(o+14)*21)]}
    expressions = ['('+'+'.join(f'"d_{i}"' for i in range(c*21+1,(c+1)*21+1))+f') AS cycle_{c}' for c in range(92)]
    m5path = (m5root/'sales_train_evaluation.csv').as_posix().replace("'","''")
    raw_m5 = con.execute('SELECT state_id warehouse,s.store_id,s.item_id,'+','.join(expressions)+
        f" FROM read_csv('{m5path}',header=true) s SEMI JOIN m5_keys k ON s.state_id=k.warehouse AND s.store_id=k.store_id AND s.item_id=k.item_id").df()
    raw_m5 = raw_m5.set_index(['warehouse','store_id','item_id'])
    m5panel = pd.read_parquet(BASE/RUNS['M5']/evidence['datasets']['M5']['panels'][-1]['source_metadata']['panel'])
    raw_m5 = raw_m5.reindex(m5panel.index)
    m5_match = bool(np.array_equal(raw_m5.to_numpy(float), m5panel.to_numpy(float)))
    evidence['m5_panel_reconciliation'] = {'series':len(raw_m5),'cycles':92,
        'max_absolute_difference':float(np.abs(raw_m5.to_numpy(float)-m5panel.to_numpy(float)).max())}
    evidence['checks'].append({'dataset':'M5','raw_quantities_match_frozen_panel':m5_match})
    for ds, d in evidence['datasets'].items():
        consumed = ['data/source/vendas/uf=BA/'+raw_ba[0].name] if ds=='BA' else ['data/m5_walmart/sales_train_evaluation.csv']
        d['consumed_raw_hash_checks'] = []
        sources = {k.replace('\\','/'):v for k,v in d['panels'][0]['source_metadata']['source_files'].items()}
        for rel in consumed:
            actual = sha(ROOT/rel)
            ok = actual == sources[rel]
            d['consumed_raw_hash_checks'].append({'file':rel,'sha256':actual,'matches_manifest':ok})
            evidence['checks'].append({'dataset':ds,'file':rel,'consumed_raw_integrity':ok})
    con.close()
    checks_ok = all(all(v for k,v in c.items() if isinstance(v,bool)) for c in evidence['checks'])
    evidence['all_integrity_checks_passed'] = checks_ok
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({'evidence_file':str(OUT/'evidence.json'),'integrity_ok':checks_ok,
        'ba_stock':evidence['stock_available_not_consumed'],
        'ba_mapping':evidence['ba_stock_join_coverage'],
        'm5_calendar':evidence['m5_real_calendar']},ensure_ascii=False,indent=2))
    if not checks_ok:
        raise SystemExit('At least one integrity check failed; inspect evidence.json')


if __name__ == '__main__':
    main()
