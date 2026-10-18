"""Isolated development pilot of frozen AIPE decisions and empirical service guards.

Fits only internal outcome selectors; never inventory controllers. All decisions
are frozen before the separate scoring phase. No distribution-free guarantee.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiment/src'))
from simulation.research.pse import CASE
from simulation.research.pse_consequences import outcome_predictions
from simulation.pipelines.demand_profiling.nodes import _series_features
from simulation.research.protocol import profile_of
from sklearn.model_selection import GroupKFold

RUN = ROOT / 'experiment/data/08_reporting/research_v2/aipe_m5_2408_unified_2026-10-15'
OUT = ROOT / 'experiment/data/service_guard_pilot/m5_four_stores_v1'
STORES = ['CA_1', 'CA_2', 'TX_1', 'WI_1']
ORIGINS = [50, 64, 78]
SIMPLE_BUFFER = .01
QUANTILE = .90


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def equal_group_quantile(values, groups, q=.9):
    """Empirical weighted quantile: each store contributes the same total mass."""
    frame = pd.DataFrame({'v': values, 'g': groups}).dropna()
    if frame.g.nunique() < 3:
        return float('inf')
    frame['w'] = 1 / frame.groupby('g').g.transform('size')
    frame = frame.sort_values('v', kind='stable')
    cumulative = frame.w.cumsum() / frame.w.sum()
    return max(0., float(frame.v.iloc[min(int(np.searchsorted(cumulative, q)), len(frame) - 1)]))


def apply_guard(rows, mode, corrections=None):
    """Reuse the candidate and cost margin; add a predeclared service veto."""
    result = rows.copy()
    current = result.policy.ne(result.reference_policy)
    delta = result.predicted_service - result.reference_predicted_service
    if mode == 'current':
        permitted = current
    elif mode == 'simple':
        permitted = current & result.predicted_service.ge(.7 + SIMPLE_BUFFER) & delta.ge(SIMPLE_BUFFER)
    elif mode == 'uncertainty':
        correction = result.candidate_policy.map(lambda p: corrections.get(p, (np.inf, np.inf)))
        absolute = np.array([v[0] for v in correction])
        differential = np.array([v[1] for v in correction])
        result['absolute_error_quantile'] = absolute
        result['differential_error_quantile'] = differential
        result['service_empirical_lower'] = result.predicted_service - absolute
        result['service_advantage_empirical_lower'] = delta - differential
        permitted = current & result.service_empirical_lower.ge(.7) & result.service_advantage_empirical_lower.ge(0)
    else:
        raise ValueError(mode)
    result['policy'] = np.where(permitted, result.candidate_policy, result.reference_policy)
    result['strategy'] = mode
    return result


def worker(origin, store, features, validation, frozen):
    """No test outcomes in arguments or fitting. Entire outer store excluded."""
    train = features[features.store_id.ne(store)]
    truth = validation[validation.store_id.ne(store)]
    assert store not in set(truth.store_id) and store not in set(train.store_id)
    outer_rows = frozen[frozen.store_id.eq(store)].copy()
    reference = outer_rows.reference_policy.unique()
    assert len(reference) == 1
    reference = reference[0]
    residuals = []
    exclusions = []
    splitter = GroupKFold(n_splits=5)
    for fit_idx, held_idx in splitter.split(train, groups=train.store_id):
        fitting, internal = train.iloc[fit_idx], train.iloc[held_idx]
        allowed = set(fitting.store_id)
        fitting_truth = truth[truth.store_id.isin(allowed)]
        assert not set(internal.store_id) & allowed
        predictions = outcome_predictions(fitting, internal, fitting_truth)
        evaluated = predictions.merge(truth, on=CASE + ['policy'], suffixes=('_pred', '_observed'), validate='one_to_one')
        baseline = evaluated[evaluated.policy.eq(reference)][CASE + ['NS_pred', 'NS_observed']].rename(
            columns={'NS_pred': 'ref_pred', 'NS_observed': 'ref_observed'})
        evaluated = evaluated.merge(baseline, on=CASE, validate='many_to_one')
        evaluated['absolute_error'] = evaluated.NS_pred - evaluated.NS_observed
        evaluated['differential_error'] = (evaluated.NS_pred - evaluated.ref_pred) - (evaluated.NS_observed - evaluated.ref_observed)
        residuals.append(evaluated)
        exclusions.append({'fitting_stores': sorted(allowed), 'internal_stores': sorted(set(internal.store_id))})
    residuals = pd.concat(residuals, ignore_index=True)
    corrections = {policy: (equal_group_quantile(g.absolute_error, g.store_id, QUANTILE),
                              equal_group_quantile(g.differential_error, g.store_id, QUANTILE))
                   for policy, g in residuals.groupby('policy')}
    rows = pd.concat([apply_guard(outer_rows, mode, corrections) for mode in ['current', 'simple', 'uncertainty']], ignore_index=True)
    assert len(rows) == 3 * len(outer_rows)
    audit = {'origin': origin, 'outer_store': store, 'reference': reference,
        'outer_training_stores': sorted(set(train.store_id)), 'inner_splits': exclusions,
        'test_used_for_calibration': False, 'inner_prediction_rows': len(residuals),
        'corrections_by_policy': {p: list(v) for p, v in corrections.items()}}
    return rows, residuals, audit


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        assert protocol['script_sha256'] == sha(Path(__file__)), 'Use a new study directory after code changes'
        return protocol
    files = [RUN / 'kpis.csv', RUN / 'selection/predictions_frozen_before_scoring.csv',
             ROOT / 'experiment/src/simulation/research/pse_consequences.py',
             ROOT / 'experiment/src/simulation/research/pse.py',
             ROOT / 'experiment/src/simulation/pipelines/demand_profiling/nodes.py',
             ROOT / 'experiment/src/simulation/research/protocol.py']
    files += [RUN / f'panels/M5_M5_{o}.parquet' for o in ORIGINS]
    protocol = {'study': 'development; all canonical test results were previously inspected',
        'stores': STORES, 'store_selection': 'first two alphabetical CA stores and first TX/WI stores; no outcome ranking',
        'origins': ORIGINS, 'inner_folds': 5, 'quantile': QUANTILE, 'simple_service_buffer': SIMPLE_BUFFER,
        'outcome_model': 'canonical ExtraTrees64/depth4/leaf3/seed20261015',
        'current_rule': 'exact frozen candidate, reference, cost margin and original policy',
        'uncertainty_rule': 'additional veto using nonnegative, store-balanced 90th percentiles of internal absolute/differential service overprediction',
        'quantile_limit': 'empirical error correction only; no calibrated coverage or noninferiority guarantee',
        'internal_difference_reference': 'fixed outer reference; residuals for all ordered candidate/reference policy pairs before selecting held candidates',
        'parameters_tuned_on_test': False, 'new_inventory_controller_fits': 0,
        'sources_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'script_sha256': sha(Path(__file__))}
    write(protocol_path, protocol)
    return protocol


def run(workers):
    protocol = prepare()
    if (OUT / 'predictions_frozen.json').exists():
        print('Frozen predictions already exist; no fits repeated.')
        return
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (ImportError, AttributeError):
        pass
    validation = pd.concat([chunk[chunk.block.eq('validation')] for chunk in pd.read_csv(
        RUN / 'kpis.csv', dtype={c: str for c in CASE[1:]}, chunksize=50000)], ignore_index=True)
    validation = validation.groupby(CASE + ['policy'], as_index=False)[['TIC', 'NS']].mean()
    frozen = pd.read_csv(RUN / 'selection/predictions_frozen_before_scoring.csv', low_memory=False)
    frozen = frozen[frozen.strategy.eq('Consequencia_recuo')]
    jobs = []
    for origin in ORIGINS:
        panel = pd.read_parquet(RUN / f'panels/M5_M5_{origin}.parquet')
        features = pd.DataFrame([dict(zip(CASE, [origin, *key])) | _series_features(values.to_numpy(float)[:origin], 17) |
            {'operational_profile': profile_of(values.to_numpy(float)[:origin])} for key, values in panel.iterrows()])
        for store in STORES:
            if not (OUT / f'fold_{origin}_{store}.json').exists():
                jobs.append((origin, store, features, validation[validation.origin.eq(origin)], frozen[frozen.origin.eq(origin)]))
    completed = 12 - len(jobs)
    write(OUT / 'status.json', {'state': 'fitting_internal_selectors', 'completed_folds': completed, 'total_folds': 12, 'workers': workers})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, *job) for job in jobs]
        for future in as_completed(futures):
            rows, residuals, audit = future.result()
            stem = f'fold_{audit["origin"]}_{audit["outer_store"]}'
            rows.to_csv(OUT / (stem + '_predictions.csv'), index=False)
            residuals.to_csv(OUT / (stem + '_internal_residuals.csv'), index=False)
            write(OUT / (stem + '.json'), audit)
            completed += 1
            write(OUT / 'status.json', {'state': 'fitting_internal_selectors', 'completed_folds': completed, 'total_folds': 12, 'workers': workers})
            print(f'{completed}/12 internal folds completed', flush=True)
    predictions = pd.concat([pd.read_csv(OUT / f'fold_{o}_{s}_predictions.csv') for o in ORIGINS for s in STORES], ignore_index=True)
    assert not predictions.duplicated(CASE + ['strategy']).any()
    path = OUT / 'predictions.csv'
    predictions.to_csv(path, index=False)
    write(OUT / 'predictions_frozen.json', {'prediction_sha256': sha(path), 'protocol_sha256': sha(OUT / 'protocol.json'),
                                          'rows': len(predictions), 'test_scoring_started': False})
    write(OUT / 'status.json', {'state': 'predictions_frozen', 'completed_folds': 12, 'total_folds': 12})


def score():
    prepare()
    frozen = json.loads((OUT / 'predictions_frozen.json').read_text())
    assert sha(OUT / 'predictions.csv') == frozen['prediction_sha256']
    assert sha(OUT / 'protocol.json') == frozen['protocol_sha256']
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in protocol['sources_sha256'].items())
    predictions = pd.read_csv(OUT / 'predictions.csv')
    raw = pd.read_csv(RUN / 'kpis.csv')
    test = raw[raw.block.eq('test')].groupby(CASE + ['policy'], as_index=False)[['TIC', 'NS']].mean()
    scores = predictions.merge(test, on=CASE + ['policy'], validate='many_to_one')
    ref = predictions[CASE + ['reference_policy']].drop_duplicates().rename(columns={'reference_policy': 'policy'}).merge(test, on=CASE + ['policy'], validate='one_to_one')
    scores = scores.merge(ref[CASE + ['TIC', 'NS']], on=CASE, suffixes=('', '_reference'), validate='many_to_one')
    scores['new_failure'] = scores.NS_reference.ge(.7 - 1e-9) & scores.NS.lt(.7 - 1e-9)
    scores['recovery'] = scores.NS_reference.lt(.7 - 1e-9) & scores.NS.ge(.7 - 1e-9)
    scores['changed'] = scores.policy.ne(scores.reference_policy)
    scores['higher_cost'] = scores.TIC.gt(scores.TIC_reference + 1e-6)
    scores['lower_service'] = scores.NS.lt(scores.NS_reference - 1e-9)
    original = pd.read_csv(RUN / 'selection/case_metrics.csv', low_memory=False)
    current = scores[scores.strategy.eq('current')].merge(original[original.strategy.eq('Consequencia_recuo')], on=CASE, suffixes=('', '_canonical'), validate='one_to_one')
    assert current.policy.eq(current.policy_canonical).all()
    np.testing.assert_allclose(current.TIC, current.TIC_canonical)
    np.testing.assert_allclose(current.NS, current.NS_canonical, equal_nan=True)
    summary = []
    for strategy, g in scores.groupby('strategy'):
        summary.append({'strategy': strategy, 'cases': len(g), 'defined_service': int(g.NS.notna().sum()),
            'mean_cost': g.TIC.mean(), 'mean_service': g.NS.mean(),
            'cost_change_percent_vs_global': 100 * (g.TIC.mean() / g.TIC_reference.mean() - 1),
            'service_change_pp_vs_global': 100 * (g.NS - g.NS_reference).mean(),
            'interventions': int(g.changed.sum()), 'new_failures': int(g.new_failure.sum()), 'recoveries': int(g.recovery.sum()),
            'higher_cost_cases': int(g.higher_cost.sum()), 'lower_service_cases': int(g.lower_service.sum())})
    scores.to_csv(OUT / 'case_metrics.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    write(OUT / 'summary.json', {'scope': 'four-store development pilot; not independent confirmation', 'results': summary})
    write(OUT / 'status.json', {'state': 'completed', 'completed_folds': 12, 'total_folds': 12, 'controller_fits': 0})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'score'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1 to 4 low-priority CPU workers for this pilot')
    run(args.workers) if args.command == 'run' else score()
