"""Read-only queue diagnostics, including measured training cost by algorithm."""
from pathlib import Path
import argparse
import json
import sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='census_2026-10-14')
    args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run name')
    output=ROOT/'experiment/data/08_reporting/research_v2'/args.run
    state=json.loads((output/'status.json').read_text(encoding='utf-8'))
    print(json.dumps(state,indent=2,ensure_ascii=False))
    database=output/'results.sqlite'
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as con:
        table=pd.read_sql_query('''SELECT dataset,uf,origin,policy,count(*) completed_fits,
            round(avg(fit_seconds),2) mean_fit_seconds,round(sum(fit_seconds),2) total_fit_seconds
            FROM fits GROUP BY dataset,uf,origin,policy ORDER BY dataset,uf,origin,policy''',con)
    print(table.to_string(index=False))
    print('Diagnostic timings and partial counts; not a completed statistical comparison.')


if __name__=='__main__':main()
