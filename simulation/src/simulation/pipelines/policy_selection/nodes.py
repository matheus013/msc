"""
policy_selection/nodes.py — Policy Selection Engine (AIPE — Fase 2).

Fluxo:
  demand_features + kpis + params → generate_policy_labels → policy_labels
  demand_features + policy_labels + params → train_policy_selector → policy_selector_model
                                                                    + policy_selector_metrics
  demand_features + policy_selector_model → apply_policy_selector → policy_recommendations

O meta-modelo é um classificador XGBoost (ou LightGBM como fallback) que,
dadas as features operacionais de uma série, recomenda qual política de
controle de inventário deve ser aplicada.

Critério de rótulo ("melhor política"):
  Minimiza TIC (Custo Total de Inventário) entre as políticas que atingem
  NS (Nível de Serviço) >= service_level_min_label.
  Se nenhuma política atinge NS mínimo, escolhe a de maior NS.
"""
import logging
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEYS = ["warehouse", "store_id", "item_id"]

FEATURE_COLS = [
    "adi", "cv2", "intermittency_ratio", "p_zero",
    "zero_streak_max", "zero_streak_mean", "burstiness",
    "demand_skewness", "demand_kurtosis", "entropy",
    "trend_coef", "seasonality_acf",
    "mu", "sigma", "cv", "n_periods", "n_positive",
]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Gera rótulos: melhor política por série
# ─────────────────────────────────────────────────────────────────────────────

def generate_policy_labels(
    kpis: pd.DataFrame,
    demand_features: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Para cada série, agrega os KPIs por política (média sobre replicações) e
    escolhe a "melhor política" como rótulo-alvo do classificador.

    Critério de seleção:
      1. Considera apenas políticas com NS >= service_level_min_label
      2. Entre essas, escolhe a que minimiza TIC
      3. Fallback: política com maior NS (se nenhuma atinge o mínimo)

    Retorna DataFrame com colunas KEYS + ['best_policy', 'best_tic', 'best_ns'].
    """
    ns_min: float = params.get("service_level_min_label", 0.70)

    # Normaliza nomes de colunas antes do groupby
    kpis = _normalize_kpi_cols(kpis)

    # Determina quais colunas canônicas existem de fato no DataFrame
    available = {c for c in kpis.columns}
    agg_cols = [c for c in ["tic", "ns", "stockout_rate"] if c in available]
    if not agg_cols:
        raise KeyError(
            f"Nenhuma coluna KPI canônica encontrada. Colunas disponíveis: {list(kpis.columns)}"
        )

    # Agrega KPIs por (warehouse, store_id, item_id, policy) — média sobre replicações
    kpi_agg = (
        kpis.groupby(KEYS + ["policy"])[agg_cols]
        .mean()
        .reset_index()
    )

    tic_col = "tic" if "tic" in kpi_agg.columns else agg_cols[0]
    ns_col  = "ns"  if "ns"  in kpi_agg.columns else None

    records = []
    for key, grp in kpi_agg.groupby(KEYS):
        if ns_col and ns_col in grp.columns:
            feasible = grp[grp[ns_col] >= ns_min]
            if len(feasible) > 0:
                best_row = feasible.loc[feasible[tic_col].idxmin()]
            else:
                best_row = grp.loc[grp[ns_col].idxmax()]
        else:
            best_row = grp.loc[grp[tic_col].idxmin()]

        rec = dict(zip(KEYS, key if isinstance(key, tuple) else (key,)))
        rec["best_policy"] = best_row["policy"]
        rec["best_tic"]    = float(best_row["tic"])
        rec["best_ns"]     = float(best_row["ns"])
        records.append(rec)

    labels = pd.DataFrame(records)

    # Merge com demand_features para ter tudo junto
    result = demand_features[KEYS].merge(labels, on=KEYS, how="inner")

    dist = result["best_policy"].value_counts().to_dict()
    log.info(
        "policy_labels: %d séries | distribuição de rótulos: %s | NS_min=%.2f",
        len(result), dist, ns_min,
    )
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Treina o meta-modelo (XGBoost / LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def train_policy_selector(
    demand_features: pd.DataFrame,
    policy_labels: pd.DataFrame,
    params: dict,
) -> Tuple[dict, pd.DataFrame]:
    """
    Treina um classificador multiclasse (XGBoost → LightGBM → RandomForest
    como fallbacks) que mapeia features operacionais → melhor política.

    Retorna:
      policy_selector_model  — dict com 'model', 'label_encoder', 'feature_cols'
      policy_selector_metrics — DataFrame com acurácia, F1 por política etc.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    import warnings

    random_state: int = params.get("random_seed", 42)
    cv_folds: int     = params.get("cv_folds", 5)
    model_type: str   = params.get("model", "xgboost")

    # Junta features + rótulos
    df = demand_features.merge(policy_labels[KEYS + ["best_policy"]], on=KEYS, how="inner")
    df = df.dropna(subset=FEATURE_COLS + ["best_policy"])

    # Remove features ausentes do dataset
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_feats].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(df["best_policy"])

    clf = _build_classifier(model_type, random_state, params)

    # Validação cruzada estratificada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            clf, X, y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=True,
        )

    # Treina modelo final em todo o dataset
    clf.fit(X, y)

    # Métricas
    y_pred = clf.predict(X)
    report = classification_report(
        y, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    metrics_rows = []
    for policy_name, m in report.items():
        if isinstance(m, dict):
            metrics_rows.append({
                "policy": policy_name,
                "precision": m.get("precision", 0),
                "recall":    m.get("recall", 0),
                "f1":        m.get("f1-score", 0),
                "support":   m.get("support", 0),
            })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["cv_accuracy_mean"] = np.mean(cv_results["test_accuracy"])
    metrics_df["cv_accuracy_std"]  = np.std(cv_results["test_accuracy"])
    metrics_df["cv_f1_macro_mean"] = np.mean(cv_results["test_f1_macro"])

    log.info(
        "Policy selector treinado | modelo=%s | CV accuracy=%.3f±%.3f | CV F1_macro=%.3f",
        model_type,
        np.mean(cv_results["test_accuracy"]),
        np.std(cv_results["test_accuracy"]),
        np.mean(cv_results["test_f1_macro"]),
    )

    model_bundle = {
        "model":         clf,
        "label_encoder": le,
        "feature_cols":  available_feats,
        "model_type":    model_type,
        "cv_results":    cv_results,
    }
    return model_bundle, metrics_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Aplica o meta-modelo a todas as séries
# ─────────────────────────────────────────────────────────────────────────────

def apply_policy_selector(
    demand_features: pd.DataFrame,
    policy_selector_model: dict,
) -> pd.DataFrame:
    """
    Aplica o modelo treinado a demand_features e retorna a política recomendada
    com as probabilidades de cada classe.

    Saída: DataFrame com KEYS + ['recommended_policy', 'confidence', prob_<policy>...]
    """
    clf    = policy_selector_model["model"]
    le     = policy_selector_model["label_encoder"]
    feats  = policy_selector_model["feature_cols"]

    # Garante que todas as features necessárias estejam presentes
    missing = [f for f in feats if f not in demand_features.columns]
    if missing:
        log.warning("Features ausentes em demand_features: %s — imputadas com 0", missing)
        for m in missing:
            demand_features = demand_features.copy()
            demand_features[m] = 0.0

    X = demand_features[feats].fillna(0.0).values.astype(float)

    y_pred = clf.predict(X)
    recommended = le.inverse_transform(y_pred)

    result = demand_features[KEYS].copy()
    result["recommended_policy"] = recommended

    # Probabilidades por classe (se suportado)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        for i, cls_name in enumerate(le.classes_):
            result[f"prob_{cls_name}"] = proba[:, i]
        result["confidence"] = proba.max(axis=1)
    else:
        result["confidence"] = 1.0

    policy_dist = result["recommended_policy"].value_counts().to_dict()
    log.info("Recomendações geradas para %d séries: %s", len(result), policy_dist)
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_kpi_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia nomes alternativos de KPIs para os canônicos usados aqui."""
    col_map = {
        # custo total
        "TIC": "tic",
        "total_inventory_cost": "tic",
        "CTI": "tic",
        # nível de serviço
        "NS": "ns",
        "ServiceLevel": "ns",
        "service_level": "ns",
        # taxa de ruptura
        "TR": "stockout_rate",
        "StockoutRate": "stockout_rate",
        "stockout_rate": "stockout_rate",
        # nome da política
        "policy_name": "policy",
    }
    return df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})


def _build_classifier(model_type: str, random_state: int, params: dict):
    """Instancia o classificador com fallback automático."""
    xgb_params = params.get("xgboost", {})
    lgb_params  = params.get("lightgbm", {})

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=xgb_params.get("n_estimators", 300),
                max_depth=xgb_params.get("max_depth", 6),
                learning_rate=xgb_params.get("learning_rate", 0.05),
                subsample=xgb_params.get("subsample", 0.8),
                colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=random_state,
                verbosity=0,
            )
        except ImportError:
            log.warning("xgboost não disponível — usando LightGBM")
            model_type = "lightgbm"

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=lgb_params.get("n_estimators", 300),
                max_depth=lgb_params.get("max_depth", 6),
                learning_rate=lgb_params.get("learning_rate", 0.05),
                random_state=random_state,
                verbose=-1,
            )
        except ImportError:
            log.warning("lightgbm não disponível — usando RandomForest")

    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
