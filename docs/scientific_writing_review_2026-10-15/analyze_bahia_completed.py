"""Read-only audit of the completed BA run; writes manuscript derivatives only."""
from pathlib import Path
import hashlib
import json
import sqlite3
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'experiment/data/08_reporting/research_v2/bahia145_cpu16_2026-10-15'
OUT = Path(__file__).resolve().parent / 'bahia_completed'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'experiment/src'))
from simulation.research.visualizations import balanced_results, held_store_selections

status = json.loads((RUN/'status.json').read_text())
manifest = json.loads((RUN/'manifest.json').read_text())
assert status['state'] == manifest['status'] == 'completed'
assert status['completed_fits'] == status['planned_fits'] == 4350
raw = pd.read_csv(RUN/'kpis.csv', dtype={c: str for c in ['warehouse','store_id','item_id']})
unit = ['dataset','origin','warehouse','store_id','item_id']
assert len(raw) == 8700 and not raw.duplicated(unit+['policy','seed','block']).any()
assert set(raw.origin) == {17,23} and set(raw.policy) == set(manifest['policies'])
assert len(raw[unit].drop_duplicates()) == 290
assert len(raw[['warehouse','store_id','item_id']].drop_duplicates()) == 145
assert (raw[['TIC','Demand','Served']].to_numpy() >= 0).all()
assert np.allclose(raw.loc[raw.Demand.gt(0),'NS'],
                   raw.loc[raw.Demand.gt(0),'Served']/raw.loc[raw.Demand.gt(0),'Demand'])
assert raw.loc[raw.Demand.eq(0),'NS'].isna().all()
db = sqlite3.connect(f'{(RUN/"results.sqlite").as_uri()}?mode=ro', uri=True)
assert db.execute('select count(*) from fits').fetchone()[0] == 4350
db.close()
balanced, incomplete = balanced_results(raw, manifest['policies'], manifest['independent_training_seeds'])
assert incomplete == 0 and len(balanced) == 4060
selections = held_store_selections(balanced, [.7,.8,.85,.9,.95])
selections.to_csv(OUT/'selections.csv', index=False)

avg = raw.groupby(unit+['profile','policy','block'], observed=True)[
    ['TIC','NS','TR','BE','FP','Demand','Served','TerminalCommittedUnits']].mean().reset_index()
test = avg[avg.block.eq('test')].copy()
selections = selections.merge(test[unit+['policy','Demand','Served']],
                              on=unit+['policy'], how='left', validate='many_to_one')
assert selections.Demand.notna().all()
selections.to_csv(OUT/'selections.csv', index=False)

def summarize(g):
    return pd.Series({'cases':len(g), 'TIC':g.TIC.mean(), 'NS':g.NS.mean(),
        'weighted_NS':g.Served.sum()/g.Demand.sum() if 'Served' in g and g.Demand.sum() else np.nan,
        'undefined_NS':g.NS.isna().sum(), 'below_070':g.NS.lt(.7).sum(),
        'TR':g.TR.mean() if 'TR' in g else np.nan,
        'FP':g.FP.mean() if 'FP' in g else np.nan,
        'fallback':g.fallback.sum() if 'fallback' in g else 0})

policy_origin = test.groupby(['origin','policy'], observed=True).apply(summarize, include_groups=False).reset_index()
policy_all = test.groupby('policy', observed=True).apply(summarize, include_groups=False).reset_index()
selection_origin = selections.groupby(['origin','alpha','strategy'], observed=True).apply(summarize, include_groups=False).reset_index()
selection_all = selections.groupby(['alpha','strategy'], observed=True).apply(summarize, include_groups=False).reset_index()
for name, frame in [('policy_by_origin',policy_origin),('policy_pooled',policy_all),
                    ('strategies_by_origin',selection_origin),('strategies_pooled',selection_all)]:
    frame.to_csv(OUT/f'{name}.csv', index=False)

# Paired conditional bootstrap of stores: preserve all origins/items/seeds of a store.
# No retraining/refitting of choices inside resamples; not a training uncertainty CI.
def conditional_interval(frame, left, right, label, group_col):
    costs = frame.pivot(index=unit, columns=group_col, values='TIC')[[left,right]].reset_index()
    clusters = costs.groupby(['warehouse','store_id'], observed=True)[[left,right]].sum()
    rng = np.random.default_rng(20261015)
    counts = rng.multinomial(len(clusters), np.full(len(clusters),1/len(clusters)), size=5000)
    totals = counts @ clusters.to_numpy()
    reduction = 100*(totals[:,0]-totals[:,1])/totals[:,0]
    point = 100*(costs[left].sum()-costs[right].sum())/costs[left].sum()
    return {'contrast':label, 'cost_reduction_percent':point,
            'conditional_CI95':np.quantile(reduction,[.025,.975]).tolist(),
            'stores':len(clusters), 'cases':len(costs)}

intervals = [conditional_interval(selections[np.isclose(selections.alpha,.7)],
             'global','perfil','global_vs_profile','strategy')]
for left,right in [('GA','GA-DQN'),('DQN','GA-DQN')]:
    intervals.append(conditional_interval(test,left,right,f'{left}_vs_{right}','policy'))
report = {'run':RUN.name, 'completed_fits':4350,'KPI_rows':8700,
    'series':145,'stores':len(raw[['warehouse','store_id']].drop_duplicates()),
    'items':raw.item_id.nunique(),'series_origin_cases':290,'origins':[17,23],
    'train_validation_test':[[17,7,7],[23,7,7]],
    'test_cycle_overlap':1, 'training_seeds':[11,29,47],
    'profiles_by_origin':avg[avg.block.eq('test')].drop_duplicates(unit).groupby(['origin','profile']).size().to_dict().__str__(),
    'selection_counts_070':selections[np.isclose(selections.alpha,.7)].groupby(['origin','strategy','policy']).size().to_dict().__str__(),
    'conditional_intervals':intervals,
    'source_sha256':{name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in ['kpis.csv','manifest.json','status.json']},
    'limitations':['Historical membership; past-only fitting does not make cohort prospective.',
       'Origins are repeated measurements and their seven-cycle tests overlap by one cycle.',
       'Seeds averaged per series/origin, not independent retail units.',
       'Intervals conditional on fitted candidates and choices; shared-product dependence remains.',
       'Fixed common baseline initial states; candidate policies do not carry their own validation stocks into test.',
       'Test ranking is descriptive and must not be reused to tune the selection rule.']}
(OUT/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
print('POLICIES\n'+policy_all.to_string(index=False))
print('STRATEGIES\n'+selection_origin.to_string(index=False))
print('POOLED\n'+selection_all.to_string(index=False))
