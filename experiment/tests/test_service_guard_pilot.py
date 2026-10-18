"""Scientific contracts for the added veto, not just implementation mirrors."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('service_guard_pilot', Path(__file__).resolve().parents[1] / 'scripts/analysis/run_service_guard_pilot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return pd.DataFrame({'policy': ['candidate', 'reference', 'candidate'],
        'candidate_policy': ['candidate'] * 3, 'reference_policy': ['reference'] * 3,
        'predicted_service': [.80, .90, .75], 'reference_predicted_service': [.76, .80, .749]})


def test_guards_cannot_reactivate_rejected_cost_intervention():
    for mode in ['current', 'simple', 'uncertainty']:
        result = module.apply_guard(rows(), mode, {'candidate': (0., 0.)})
        assert result.policy.iloc[1] == 'reference'


def test_uncertainty_veto_covers_absolute_and_paired_service_errors():
    assert module.apply_guard(rows(), 'uncertainty', {'candidate': (.11, 0.)}).policy.eq('reference').all()
    assert module.apply_guard(rows(), 'uncertainty', {'candidate': (0., .05)}).policy.eq('reference').all()
    result = module.apply_guard(rows(), 'uncertainty', {'candidate': (.02, .01)})
    assert result.policy.tolist() == ['candidate', 'reference', 'reference']


def test_empirical_quantile_requires_multiple_stores_and_equal_store_mass():
    assert np.isinf(module.equal_group_quantile([.1, .2], ['A', 'B']))
    # A large store cannot overwhelm the two small stores in calibration.
    values = [0.] * 100 + [.2, .3]
    groups = ['A'] * 100 + ['B', 'C']
    assert module.equal_group_quantile(values, groups, .9) == .3
    assert module.equal_group_quantile([-.3, -.2, -.1], ['A', 'B', 'C']) == 0


def test_missing_policy_calibration_falls_back_to_reference():
    assert module.apply_guard(rows(), 'uncertainty', {}).policy.eq('reference').all()
