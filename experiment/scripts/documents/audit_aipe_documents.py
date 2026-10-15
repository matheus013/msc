"""Reconcile documentary claims with completed BA/M5 matrices, without fitting."""
from pathlib import Path
import hashlib
import json
import sqlite3
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
from simulation.research.pse import CASE,score_predictions

BASE=ROOT/'experiment/data/08_reporting/research_v2'
ARTICLE=ROOT/'docs/supply_chain_analytics_article'
TARGET=ARTICLE/'reviews/aipe_revision_2026-10-15'
TARGET.mkdir(parents=True,exist_ok=True)
sources={}


def read(path):
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,dtype={k:str for k in CASE[1:]})


def fits(name,expected):
    path=BASE/name/'results.sqlite'
    db=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    count=db.execute('SELECT count(*) FROM fits').fetchone()[0];db.close()
    assert count==expected
    return count


ba=BASE/'bahia145_cpu16_2026-10-15'
raw=read(ba/'kpis.csv')
test=raw[raw.block.eq('test')].groupby(CASE+['policy'],as_index=False)[['TIC','NS']].mean()
selection=read(ROOT/'docs/scientific_writing_review_2026-10-15/bahia_completed/selections.csv')
selection=selection[np.isclose(selection.alpha,.7)&selection.block.eq('test')]
assert len(selection)==580
matched=selection.merge(test,on=CASE+['policy'],suffixes=('_published','_matrix'),validate='many_to_one')
assert len(matched)==580
np.testing.assert_allclose(matched.TIC_published,matched.TIC_matrix,atol=1e-8)
np.testing.assert_allclose(matched.NS_published,matched.NS_matrix,atol=1e-8,equal_nan=True)
means=selection.groupby('strategy')[['TIC','NS']].mean()
cost=100*(means.loc['perfil','TIC']/means.loc['global','TIC']-1)
service=100*(means.loc['perfil','NS']-means.loc['global','NS'])
np.testing.assert_allclose([cost,service],[1.9259705457970908,3.8869702985026677],atol=1e-8)
assert int(selection[selection.strategy.eq('global')].NS.lt(.7).sum())==119
assert int(selection[selection.strategy.eq('perfil')].NS.lt(.7).sum())==100
old_audit=json.loads((ARTICLE/'reviews/fine_review_2026-10-15/numerical_audit.json').read_text())
for path,digest in old_audit['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest

m5=BASE/'m5_pse_pilot30_v2_2026-10-15'
raw5=read(m5/'kpis.csv');predictions=read(m5/'predictions_frozen_before_test.csv')
guard=read(m5/'diagnostics_v1/guarded_predictions_before_scoring.csv')
scored=score_predictions(pd.concat([predictions,guard[predictions.columns]],ignore_index=True),raw5[raw5.block.eq('test')])
original=json.loads((m5/'summary.json').read_text())
for name,group in scored.groupby('strategy'):
    if name not in original['strategies']:continue
    reference=original['strategies'][name]
    np.testing.assert_allclose([group.TIC.mean(),group.NS.mean()],[reference['mean_cost'],reference['mean_service']],atol=1e-9)
    assert int(group.oracle_hit.sum())==reference['oracle_hits']
    assert int(group.feasible.sum())==reference['feasible_cases']
assert scored.groupby('strategy').size().eq(90).all()
assert scored.groupby('strategy').NS.count().eq(86).all()
assert (scored.loc[scored.comparable,'efficiency_percent']<=100+1e-6).all()
means5=scored.groupby('strategy')[['TIC','NS']].mean()
cost5=100*(means5.loc['AIPE_PSE','TIC']/means5.loc['Global','TIC']-1)
service5=100*(means5.loc['AIPE_PSE','NS']-means5.loc['Global','NS'])
reduction=100*(1-means5.loc['PSE_com_recuo','TIC']/means5.loc['AIPE_PSE','TIC'])
text=(ARTICLE/'m5_pse_20261015.tex').read_text(encoding='utf-8')
for name,group in scored.groupby('strategy'):
    formatted=f'{group.TIC.mean():,.2f}'.translate(str.maketrans(',.','.,'))
    assert formatted in text,formatted
assert f'{cost5:.1f}'.replace('.',',') in (ARTICLE/'manuscript_ptbr.tex').read_text(encoding='utf-8')
assert (ARTICLE/'m5_pse_20261015.tex').read_bytes()==(ROOT/'docs/master_proposal/m5_pse_20261015.tex').read_bytes()
plans=json.loads((m5/'perfect_information.json').read_text())
assert len(plans)==90 and all(p['proven_optimal'] for p in plans)
failure=read(m5/'diagnostics_v1/service_failure_decomposition.csv').set_index('strategy')
assert failure.loc['AIPE_PSE',['atendeu','erro_selecao','lacuna_portfolio','limite_fisico','demanda_zero']].tolist()==[39,21,15,11,4]
report={'status':'passed','inventory_simulations_trained':0,
    'Bahia':{'fits':fits(ba.name,4350),'selections_reconciled':580,'cost_change_percent':cost,'service_change_pp':service,
        'source_matrices_unchanged_from_prior_audit':True},
    'M5':{'fits':fits(m5.name,1350),'series':30,'stores':10,'series_origin_cases':90,'defined_service_cases':86,
        'original_strategies_reconciled':5,'guarded_strategy_exploratory':True,'predictive_cost_change_percent':cost5,
        'predictive_service_change_pp':service5,'fallback_cost_reduction_percent':reduction,
        'proven_perfect_information_optima':90,'failure_partition':[39,21,15,11,4]},
    'shared_M5_text_identical':True,'input_sha256':sources,
    'scope':'Numerical reconciliation of completed matrices; not independent confirmation or replication of policy training.'}
(TARGET/'numerical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))
