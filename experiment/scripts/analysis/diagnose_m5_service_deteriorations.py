"""Read-only, post-hoc diagnosis of the frozen M5 guarded selector.

No fitting, threshold optimization, simulation, or canonical-file mutation.
Outputs describe associations; they do not validate a new intervention rule.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiment/src'))
from simulation.pipelines.demand_profiling.nodes import _series_features
from simulation.research.protocol import profile_of

RUN = ROOT / 'experiment/data/08_reporting/research_v2/aipe_m5_2408_unified_2026-10-15'
OUT = ROOT / 'docs/operations/m5_service_deteriorations_2026-10-18'
CASE = ['origin', 'warehouse', 'store_id', 'item_id']
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    return pd.read_csv(path, dtype={c: str for c in CASE[1:]}, low_memory=False)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for source in ['experiment/src/simulation/pipelines/demand_profiling/nodes.py',
                   'experiment/src/simulation/research/protocol.py']:
        SOURCES[source] = digest(ROOT / source)
    scored = read(RUN / 'selection/case_metrics.csv')
    frozen = read(RUN / 'selection/predictions_frozen_before_scoring.csv')
    raw = read(RUN / 'kpis.csv')
    reference = scored[scored.strategy.eq('Global')][CASE + ['policy', 'TIC', 'NS']]
    paired = scored[scored.strategy.eq('Consequencia_recuo')].merge(
        reference, on=CASE, suffixes=('', '_reference'), validate='one_to_one')
    f = frozen[frozen.strategy.eq('Consequencia_recuo')]
    check = paired.merge(f[CASE + ['policy', 'predicted_service', 'predicted_cost']],
                         on=CASE, suffixes=('', '_frozen'), validate='one_to_one')
    assert check.policy.eq(check.policy_frozen).all()
    np.testing.assert_allclose(check.predicted_service, check.predicted_service_frozen)
    np.testing.assert_allclose(check.predicted_cost, check.predicted_cost_frozen)
    means = raw.groupby(CASE + ['policy', 'block'], as_index=False)[['TIC', 'NS', 'Demand', 'Served']].mean()
    for side in ['', '_reference']:
        observed = paired.merge(means[means.block.eq('test')],
            left_on=CASE + ['policy' + side], right_on=CASE + ['policy'], suffixes=('', '_checked'), validate='many_to_one')
        np.testing.assert_allclose(observed['TIC' + side], observed.TIC_checked)
        np.testing.assert_allclose(observed['NS' + side], observed.NS_checked, equal_nan=True)
    paired['changed'] = paired.policy.ne(paired.policy_reference)
    paired['reference_feasible'] = paired.NS_reference.ge(.7 - 1e-9)
    paired['new_failure'] = paired.reference_feasible & paired.NS.lt(.7 - 1e-9)
    paired['recovery'] = paired.NS_reference.lt(.7 - 1e-9) & paired.NS.ge(.7 - 1e-9)
    paired['saving'] = paired.TIC_reference - paired.TIC
    paired['service_loss_pp'] = 100 * (paired.NS_reference - paired.NS)
    paired['candidate_service_overprediction_pp'] = 100 * (paired.predicted_service - paired.NS)
    paired['reference_service_overprediction_pp'] = 100 * (paired.reference_predicted_service - paired.NS_reference)
    paired['predicted_service_advantage_pp'] = 100 * (paired.predicted_service - paired.reference_predicted_service)
    paired['predicted_service_buffer_pp'] = 100 * (paired.predicted_service - .7)
    paired['predicted_cost_saving_percent'] = 100 * (1 - paired.predicted_cost / paired.reference_predicted_cost)
    assert len(paired) == 5326 and paired.changed.sum() == 679
    assert paired.new_failure.sum() == 19 and paired.recovery.sum() == 43

    validation = means[means.block.eq('validation')]
    for policy_column, prefix in [('policy', 'candidate_validation'), ('policy_reference', 'reference_validation')]:
        values = validation[CASE + ['policy', 'TIC', 'NS']].rename(columns={
            'policy': policy_column, 'TIC': prefix + '_cost', 'NS': prefix + '_service'})
        paired = paired.merge(values, on=CASE + [policy_column], validate='many_to_one')
    paired['validation_service_advantage_pp'] = 100 * (paired.candidate_validation_service - paired.reference_validation_service)

    feature_rows = []
    for origin, cohort in paired.groupby('origin'):
        path = RUN / 'panels' / f'M5_M5_{origin}.parquet'
        SOURCES[str(path.relative_to(ROOT))] = digest(path)
        panel = pd.read_parquet(path)
        for row in cohort.itertuples():
            demand = panel.loc[(row.warehouse, row.store_id, row.item_id)].to_numpy(float)
            prefix = demand[:origin]
            val = demand[origin:origin + 7]
            test = demand[origin + 7:origin + 14]
            features = _series_features(prefix, 17)
            feature_rows.append(dict(zip(CASE, (origin, row.warehouse, row.store_id, row.item_id))) | features | {
                'operational_profile': profile_of(prefix),
                'prefix_last7_mean': prefix[-7:].mean(),
                'validation_demand_mean': val.mean(),
                'test_demand_mean_retrospective': test.mean(),
                'test_to_validation_demand_ratio_retrospective': test.mean() / val.mean() if val.mean() else np.nan})
    paired = paired.merge(pd.DataFrame(feature_rows), on=CASE, validate='one_to_one')
    failures = paired[paired.new_failure].copy().sort_values('service_loss_pp', ascending=False)
    assert failures.policy.eq('sSCycle').all()

    failure_seed_rows = []
    for row in failures.itertuples():
        mask = np.logical_and.reduce([raw[c].eq(getattr(row, c)).to_numpy() for c in CASE]) & raw.block.eq('test').to_numpy()
        g = raw.loc[mask]
        for side, policy in [('candidate', row.policy), ('reference', row.policy_reference)]:
            for v in g[g.policy.eq(policy)].itertuples():
                failure_seed_rows.append(dict(zip(CASE, [getattr(row, c) for c in CASE])) |
                    {'side': side, 'policy': policy, 'seed': v.seed, 'cost': v.TIC,
                     'service': v.NS, 'service_feasible': v.NS >= .7 - 1e-9})
    pd.DataFrame(failure_seed_rows).to_csv(OUT / 'failure_policy_seed_metrics.csv', index=False)
    paired.to_csv(OUT / 'all_cases_diagnostic.csv', index=False)
    failures.to_csv(OUT / 'nineteen_new_failures.csv', index=False)

    groups = paired[paired.changed].groupby(['origin', 'policy_reference', 'policy'], dropna=False).agg(
        interventions=('changed', 'size'), reference_feasible=('reference_feasible', 'sum'),
        new_failures=('new_failure', 'sum'), recoveries=('recovery', 'sum'), total_saving=('saving', 'sum'))
    groups['new_failure_rate_given_feasible_reference'] = groups.new_failures / groups.reference_feasible.replace(0, np.nan)
    groups.to_csv(OUT / 'transition_denominators.csv')
    signal_columns = ['predicted_service_buffer_pp', 'predicted_service_advantage_pp',
        'predicted_cost_saving_percent', 'mu', 'cv2', 'adi', 'p_zero',
        'zero_streak_max', 'trend_coef', 'validation_service_advantage_pp']
    signal_rows = []
    for name, data in [('new_failures', failures), ('other_interventions_with_feasible_reference',
            paired[paired.changed & paired.reference_feasible & ~paired.new_failure]),
            ('all_interventions', paired[paired.changed])]:
        for column in signal_columns:
            v = data[column].dropna()
            signal_rows.append({'group': name, 'signal': column, 'n': len(v), 'median': v.median(),
                'q25': v.quantile(.25), 'q75': v.quantile(.75), 'minimum': v.min(), 'maximum': v.max()})
    pd.DataFrame(signal_rows).to_csv(OUT / 'predecision_signal_descriptions.csv', index=False)
    aggregate_saving = paired.saving.sum()
    payload = {'analysis': 'post-hoc descriptive diagnosis; no fitting or rule changes',
        'cases': len(paired), 'interventions': int(paired.changed.sum()), 'new_failures': len(failures),
        'recoveries': int(paired.recovery.sum()), 'failure_unique_series': len(failures[CASE[1:]].drop_duplicates()),
        'interventions_with_feasible_reference': int((paired.changed & paired.reference_feasible).sum()),
        'new_failure_rate_given_intervention_and_feasible_reference': float(len(failures) / (paired.changed & paired.reference_feasible).sum()),
        'failure_stores': int(failures.store_id.nunique()),
        'failure_origins': failures.origin.value_counts().to_dict(),
        'failure_reference_policies': failures.policy_reference.value_counts().to_dict(),
        'failure_profiles': failures.operational_profile.value_counts().to_dict(),
        'failure_higher_cost_cases': int(failures.saving.lt(-1e-6).sum()),
        'failure_positive_saving_cases': int(failures.saving.gt(1e-6).sum()),
        'failure_net_saving_scenario_units': failures.saving.sum(),
        'all_cases_net_saving_scenario_units': aggregate_saving,
        'failure_share_of_net_saving_percent': 100 * failures.saving.sum() / aggregate_saving,
        'service_loss_pp': failures.service_loss_pp.describe().to_dict(),
        'candidate_service_overprediction_pp': failures.candidate_service_overprediction_pp.describe().to_dict(),
        'reference_service_overprediction_pp': failures.reference_service_overprediction_pp.describe().to_dict(),
        'predicted_service_advantage_pp': failures.predicted_service_advantage_pp.describe().to_dict(),
        'candidate_failed_same_series_validation_cases': int(failures.candidate_validation_service.lt(.7 - 1e-9).sum()),
        'candidate_worse_same_series_validation_cases': int(failures.validation_service_advantage_pp.lt(-1e-7).sum()),
        'sources_sha256': SOURCES,
        'script_sha256': digest(Path(__file__)),
        'limits': ['Costs are scenario units, not BRL.', '19 cases were selected by observed test outcomes.',
            'Repeated products/stores/origins are not independent observations.',
            'Same-series validation is operationally prior but held-store outcomes were deliberately excluded from fitting; using them changes the information protocol.',
            'Group descriptions and potential warning signs are not calibrated risk scores or independently validated rules.',
            'Test demand and actual prediction errors are retrospective and cannot be predecision inputs.']}
    for relative, expected in SOURCES.items():
        assert digest(ROOT / relative) == expected, f'Input modified: {relative}'
    payload['source_hashes_unchanged_after_analysis'] = True
    (OUT / 'evidence.json').write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = np.where(failures.saving.ge(0), '#198575', '#b05238')
    axes[0].scatter(100 * failures.NS_reference, 100 * failures.NS, c=colors, edgecolor='black', linewidth=.4)
    axes[0].axhline(70, color='gray', linestyle='--')
    axes[0].set(xlabel='Atendimento da referência (%)', ylabel='Atendimento após a troca (%)', title='19 novas violações no teste')
    axes[1].barh(np.arange(len(failures)), failures.saving, color=colors)
    axes[1].set_yticks(np.arange(len(failures)), [f'{r.store_id} · {r.item_id} · {r.origin}' for r in failures.itertuples()], fontsize=6)
    axes[1].axvline(0, color='gray', linewidth=.8)
    axes[1].set(xlabel='Economia realizada na simulação (unidades de cenário)', title='Custo nos mesmos casos')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / ('nineteen_failures.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(clean({k: v for k, v in payload.items() if k not in ['sources_sha256', 'limits']}), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
