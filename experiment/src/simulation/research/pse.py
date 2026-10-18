"""Pilot PSE: per-origin leave-one-store-out; validation labels, frozen test.

All preprocessing is fitted inside each outer fold. Test KPIs are only read
after predictions. Rare classes and zero-demand cases are retained.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from simulation.pipelines.policy_selection.nodes import FEATURE_COLS
from simulation.research.oracles import efficiency_percent

CASE=['origin','warehouse','store_id','item_id']


def feature_transform(with_profile=True):
    """Fold-local feature contract shared by classification and regression."""
    transforms=[('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True)),FEATURE_COLS)]
    if with_profile:
        transforms.append(('profile',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['operational_profile']))
    return ColumnTransformer(transforms)


def fit_predict_fold(train,held,with_profile=True):
    """Fit one fold and expose scores/importances for auditable diagnostics.

    Class scores are not calibrated probabilities of operational success.
    """
    encoder=LabelEncoder().fit(train.label)
    learner=DummyClassifier(strategy='most_frequent') if len(encoder.classes_)==1 else XGBClassifier(
        n_estimators=200,max_depth=3,learning_rate=.05,subsample=1,colsample_bytree=1,
        n_jobs=1,random_state=20261015,verbosity=0)
    preprocessing=feature_transform(with_profile)
    model=make_pipeline(preprocessing,learner)
    model.fit(train,encoder.transform(train.label))
    prediction=encoder.inverse_transform(model.predict(held).astype(int))
    scores=np.asarray(model.predict_proba(held))
    names=preprocessing.get_feature_names_out()
    importance=getattr(learner,'feature_importances_',np.zeros(len(names)))
    return {'policy':prediction,'max_class_score':scores.max(axis=1),
            'class_support':np.array([int(train.label.eq(p).sum()) for p in prediction]),
            'feature_importance':dict(zip(names,importance.astype(float))),
            'model':model,'classes':encoder.classes_}


def best_policy(frame, alpha=.7):
    """Mean metrics per policy; undefined service never becomes 100%."""
    mean=frame.groupby('policy',sort=True)[['TIC','NS']].mean().reset_index()
    feasible=mean[mean.NS.ge(alpha-1e-9)]
    if len(feasible):return feasible.sort_values(['TIC','policy']).iloc[0].policy
    return mean.sort_values(['NS','TIC','policy'],ascending=[False,True,True],na_position='last').iloc[0].policy


def predict_loso(features, validation, alpha=.7):
    """No test argument by design; one prediction per case and strategy."""
    rows=[]
    labels=validation.groupby(CASE).apply(lambda g:best_policy(g,alpha),include_groups=False)
    data=features.set_index(CASE).join(labels.rename('label')).reset_index()
    if data.label.isna().any():raise ValueError('Missing validation label')
    for origin,cohort in data.groupby('origin'):
        if cohort.store_id.nunique()<2:raise ValueError('At least two stores required')
        for store in sorted(cohort.store_id.unique()):
            train=cohort[cohort.store_id.ne(store)];held=cohort[cohort.store_id.eq(store)]
            valid_train=validation[validation.origin.eq(origin)&validation.store_id.ne(store)]
            global_policy=best_policy(valid_train,alpha)
            mode=train.label.value_counts().sort_index().sort_values(ascending=False,kind='stable').index[0]
            chosen={}
            for strategy,with_profile in [('AIPE_PSE',True),('PSE_sem_perfil',False)]:
                chosen[strategy]=fit_predict_fold(train,held,with_profile)['policy']
            for pos,(_,case) in enumerate(held.iterrows()):
                profile_train=valid_train.merge(train[CASE+['operational_profile']],on=CASE)
                profile_train=profile_train[profile_train.operational_profile.eq(case.operational_profile)]
                selections={s:v[pos] for s,v in chosen.items()}
                selections.update(Global=global_policy,Perfil=best_policy(profile_train,alpha) if len(profile_train) else global_policy,Modo_rotulos=mode)
                for strategy,policy in selections.items():
                    rows.append({k:case[k] for k in CASE}|{'strategy':strategy,'policy':policy,
                        'validation_label':case.label,'label_unseen_in_training':case.label not in set(train.label)})
    return pd.DataFrame(rows)


def score_predictions(predictions,test,alpha=.7):
    mean=test.groupby(CASE+['policy'],as_index=False)[['TIC','NS']].mean()
    scored=predictions.merge(mean,on=CASE+['policy'],validate='many_to_one')
    oracle=[]
    for key,group in mean.groupby(CASE):
        name=best_policy(group,alpha);row=group[group.policy.eq(name)].iloc[0]
        oracle.append(dict(zip(CASE,key))|{'oracle_policy':name,'oracle_cost':row.TIC,'oracle_service':row.NS})
    scored=scored.merge(pd.DataFrame(oracle),on=CASE,validate='many_to_one')
    scored['feasible']=scored.NS.ge(alpha-1e-9)
    scored['comparable']=scored.feasible&scored.oracle_service.ge(alpha-1e-9)
    scored['oracle_hit']=scored.comparable&np.isclose(scored.TIC,scored.oracle_cost,rtol=1e-6,atol=1e-8)
    scored['efficiency_percent']=[efficiency_percent(r.oracle_cost,r.TIC) if r.comparable else np.nan for r in scored.itertuples()]
    scored['within_5_percent']=scored.comparable&scored.TIC.le(scored.oracle_cost*1.05+1e-8)
    scored['within_1_percent']=scored.comparable&scored.TIC.le(scored.oracle_cost*1.01+1e-8)
    return scored


def paired_store_intervals(scored,draws=2000):
    """Paired store-cluster bootstrap; keeps origins/items together (pilot CI)."""
    base=scored[scored.strategy.eq('AIPE_PSE')]
    rows=[];rng=np.random.default_rng(20261015)
    for strategy,group in scored.groupby('strategy'):
        if strategy=='AIPE_PSE':continue
        pair=base.merge(group,on=CASE,suffixes=('_aipe','_baseline'),validate='one_to_one')
        pair['cost_difference']=pair.TIC_aipe-pair.TIC_baseline
        pair['service_difference']=pair.NS_aipe-pair.NS_baseline
        stores=sorted(pair.store_id.unique())
        for metric in ['cost_difference','service_difference']:
            aggregates=pair.groupby('store_id')[metric].agg(['sum','count']).reindex(stores)
            indices=rng.integers(0,len(stores),size=(draws,len(stores)))
            numerator=aggregates['sum'].to_numpy()[indices].sum(axis=1)
            denominator=aggregates['count'].to_numpy()[indices].sum(axis=1)
            samples=numerator[denominator>0]/denominator[denominator>0]
            rows.append({'baseline':strategy,'metric':metric,'mean_difference':float(pair[metric].mean()),
                'ci95_low':float(np.quantile(samples,.025)) if len(samples) else None,
                'ci95_high':float(np.quantile(samples,.975)) if len(samples) else None,
                'store_clusters':len(stores),'bootstrap_draws':draws,'exploratory':True})
    return rows
