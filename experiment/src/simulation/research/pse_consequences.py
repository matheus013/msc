"""Development experiment: predict operational outcomes, then decide.

Functions receive validation outcomes only. No test metrics enter fitting,
inner threshold selection, or recommendations. Scores are not guarantees.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from simulation.research.pse import CASE, best_policy, feature_transform

MARGINS = (0.0, .02, .05, .10, float('inf'))


def matched_classifier(train, held, validation):
    """Same forest regularization as outcome models; winner-label control."""
    labels=validation.groupby(CASE).apply(best_policy,include_groups=False)
    target=labels.reindex(pd.MultiIndex.from_frame(train[CASE])).to_numpy()
    model=make_pipeline(feature_transform(),
        ExtraTreesClassifier(n_estimators=64,max_depth=4,min_samples_leaf=3,random_state=20261015,n_jobs=1))
    model.fit(train,target)
    rows=held[CASE].copy()
    rows['policy']=model.predict(held)
    rows['strategy']='Classe_floresta_controle'
    return rows


def outcome_predictions(train, held, validation, with_profile=True):
    """Regularized, fixed forests; log cost and service modeled separately."""
    data = validation.groupby(CASE + ['policy'])[['TIC', 'NS']].mean()
    cost = data.TIC.unstack('policy').sort_index(axis=1)
    service = data.NS.unstack('policy').reindex(columns=cost.columns)
    policies = list(cost.columns)
    index = pd.MultiIndex.from_frame(train[CASE])
    cost = cost.reindex(index)
    service = service.reindex(index)
    if cost.isna().any().any():
        raise ValueError('Incomplete validation policy matrix')
    preprocessing = feature_transform(with_profile)
    x = preprocessing.fit_transform(train)
    xh = preprocessing.transform(held)
    options = dict(n_estimators=64, max_depth=4, min_samples_leaf=3,
                   random_state=20261015, n_jobs=1)
    cm = ExtraTreesRegressor(**options).fit(x, np.log1p(cost.to_numpy()))
    valid = service.notna().all(axis=1).to_numpy()
    if not valid.any():
        raise ValueError('No defined validation service for fitting')
    sm = ExtraTreesRegressor(**options).fit(x[valid], service.to_numpy()[valid])
    cp = np.maximum(0, np.expm1(cm.predict(xh)))
    sp = np.clip(sm.predict(xh), 0, 1)
    records = []
    for pos, (_, row) in enumerate(held.iterrows()):
        for j, policy in enumerate(policies):
            records.append({k: row[k] for k in CASE} | {
                'policy': policy, 'TIC': cp[pos, j], 'NS': sp[pos, j]})
    return pd.DataFrame(records)


def decisions(predicted, reference, margin=0.0, guarded=False):
    """Override a fixed reference only for predicted feasible, cheaper action."""
    records = []
    for key, group in predicted.groupby(CASE):
        selected = best_policy(group)
        candidate = group[group.policy.eq(selected)].iloc[0]
        baseline = group[group.policy.eq(reference)].iloc[0]
        intervene = bool(np.isfinite(margin) and candidate.NS >= .7 - 1e-9
                         and candidate.NS >= baseline.NS - 1e-9
                         and candidate.TIC < baseline.TIC * (1-margin) - 1e-9)
        policy = selected if not guarded or intervene else reference
        records.append(dict(zip(CASE, key)) | {'policy': policy,
            'reference_policy': reference, 'candidate_policy': selected,
            'intervened': policy != reference, 'margin': margin,
            'predicted_cost': candidate.TIC, 'predicted_service': candidate.NS,
            'reference_predicted_cost': baseline.TIC,
            'reference_predicted_service': baseline.NS})
    return pd.DataFrame(records)


def tune_margin(inner_predictions, validation):
    """Choose lowest inner cost without reducing mean service vs reference.

    Infinity is the explicit always-reference candidate. This is empirical
    internal validation, not a calibrated or distribution-free safety claim.
    """
    truth = validation.groupby(CASE + ['policy'], as_index=False)[['TIC', 'NS']].mean()
    rows = []
    for margin in MARGINS:
        choices = pd.concat([decisions(p, reference, margin, True)
                             for p, reference in inner_predictions], ignore_index=True)
        measured = choices.merge(truth, on=CASE+['policy'], validate='one_to_one')
        baseline = choices[CASE+['reference_policy']].rename(columns={'reference_policy':'policy'}).merge(
            truth, on=CASE+['policy'], validate='one_to_one')
        rows.append({'margin':margin, 'cost':float(measured.TIC.mean()),
                     'service':float(measured.NS.mean()), 'reference_service':float(baseline.NS.mean()),
                     'interventions':int(choices.intervened.sum())})
    candidates = [r for r in rows if r['service'] >= r['reference_service']-1e-9]
    if not candidates:
        return float('inf'), rows  # Undefined validation service cannot justify intervention.
    winner = min(candidates, key=lambda r:(r['cost'], -r['service'], -r['margin']))
    return winner['margin'], rows


def predict_consequences(features, validation, progress=None, inner_folds=None):
    if inner_folds is not None and inner_folds<2:
        raise ValueError('At least two inner group folds required')
    predictions, diagnostics = [], []
    for origin, cohort in features.groupby('origin'):
        valid = validation[validation.origin.eq(origin)]
        stores = sorted(cohort.store_id.unique())
        if len(stores) < 3:
            raise ValueError('Nested store validation requires at least three stores')
        cache = {}
        for store in stores:
            train = cohort[cohort.store_id.ne(store)]
            held = cohort[cohort.store_id.eq(store)]
            outer_valid = valid[valid.store_id.ne(store)]
            reference = best_policy(outer_valid)
            predictions.append(matched_classifier(train,held,outer_valid))
            inner = []
            if inner_folds is None:
                groups = [[s] for s in sorted(train.store_id.unique())]
            else:
                splits=GroupKFold(n_splits=min(inner_folds,train.store_id.nunique()))
                groups=[sorted(train.iloc[idx].store_id.unique()) for _,idx in splits.split(train,groups=train.store_id)]
            inner_sizes=[]
            for group in groups:
                excluded = tuple(sorted([store,*group]))
                if excluded not in cache:
                    subset = cohort[~cohort.store_id.isin(excluded)]
                    targets = cohort[cohort.store_id.isin(excluded)]
                    outcomes = valid[~valid.store_id.isin(excluded)]
                    cache[excluded] = (outcome_predictions(subset, targets, outcomes), best_policy(outcomes))
                pp, ref = cache[excluded]
                inner.append((pp[pp.store_id.isin(group)], ref))
                inner_sizes.append(int((~cohort.store_id.isin(excluded)).sum()))
            margin, scores = tune_margin(inner, outer_valid)
            predicted = outcome_predictions(train, held, outer_valid)
            for strategy, guarded in [('Consequencia',False), ('Consequencia_recuo',True)]:
                rows = decisions(predicted, reference, margin, guarded)
                rows['strategy'] = strategy
                predictions.append(rows)
            without = outcome_predictions(train, held, outer_valid, with_profile=False)
            rows = decisions(without, reference)
            rows['strategy'] = 'Consequencia_sem_perfil'
            predictions.append(rows)
            diagnostics.append({'origin':int(origin), 'held_store':store,
                'reference':reference, 'margin':margin, 'inner_scores':scores,
                'outer_training_cases':len(train), 'inner_training_cases_min':min(inner_sizes),
                'inner_training_cases_max':max(inner_sizes),'inner_folds':len(groups)})
            if progress:
                progress(origin, store, margin)
    return pd.concat(predictions, ignore_index=True), diagnostics
