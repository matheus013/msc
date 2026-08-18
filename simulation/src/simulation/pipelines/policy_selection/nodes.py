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

# 2026-08-18: pedido do usuario -- "o AIPE e uma selecao de politica para
# cada perfil de loja, nao faz sentido nao usar perfil como entrada". O
# perfil operacional (POD) e categorico e DERIVADO dos FEATURE_COLS acima
# por um conjunto de regras de limiar (classify_operational_profiles) --
# em principio um classificador flexivel (XGBoost) poderia redescobrir
# essas fronteiras sozinho a partir dos FEATURE_COLS crus. Na pratica,
# arvores de decisao lidam melhor com uma categoria explicita do que com a
# tarefa de reconstruir os MESMOS cortes (ex.: "adi>=1,32 E cv2>=0,49 E
# streak>=5") via varios splits contínuos -- por isso oferecer o perfil
# como feature direta pode ajudar. `evaluate_profile_feature_gain` mede
# isso explicitamente (com vs. sem), em vez de assumir.
PROFILE_COL = "operational_profile"


def _onehot_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot do perfil operacional; devolve (df com colunas extras, nomes das colunas)."""
    dummies = pd.get_dummies(df[PROFILE_COL], prefix="profile")
    cols = list(dummies.columns)
    return pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1), cols


def _build_cv(y: np.ndarray, params: dict):
    """
    2026-08-18: CORRIGIDO (3a vez, a pedido do usuário: "não tem uma outra
    estratégia sem ser kfold?"). `StratifiedKFold` exige um mínimo de
    exemplos por classe pra dividir em treino/teste -- com 11 políticas
    candidatas e só 145 séries na Bahia, várias classes têm 1-2 exemplos
    (`CappedBaseStock`: 1), o que quebra qualquer k-fold estratificado,
    não importa quantos folds (chegava a dar `ValueError`/NaN mesmo já
    excluindo classes raras e re-densificando rótulos).

    `cv_strategy` (`params["cv_strategy"]`, default `"loo"`):
      "loo"             -> Leave-One-Out: cada fold deixa 1 série de fora.
                           NÃO exige mínimo por classe (não estratifica) --
                           usa o máximo de dado possível pra treino, o que
                           importa muito com só 145 amostras.
      "stratified_kfold" -> k-fold estratificado (comportamento anterior).

    2026-08-18 (correção real, 4a rodada): classes com < 2 exemplos são
    SEMPRE excluídas da CV, em QUALQUER estratégia -- inclusive LOO. Não é
    uma limitação de estratificação (LOO nunca estratificou); é que o
    XGBoost REJEITA um fold de treino que não contenha TODAS as classes
    vistas globalmente ("Invalid classes inferred from unique values of
    y"). Com uma classe de 1 exemplo só (`CappedBaseStock` na Bahia), o
    fold do LOO em que ela é a única deixada de fora tem ZERO exemplos
    dela no treino -- quebra sempre, não importa o esquema de split. O
    modelo final (`clf.fit(X, y)`, fora desta função) ainda treina com
    TODOS os dados, incluindo essas classes raras -- só não são avaliadas
    quantitativamente pela CV.

    Retorna (máscara booleana de linhas a manter pra CV, objeto splitter).
    """
    strategy = str(params.get("cv_strategy", "loo"))

    classes, counts = np.unique(y, return_counts=True)
    rare = classes[counts < 2]
    mask = ~np.isin(y, rare) if rare.size else np.ones(len(y), dtype=bool)
    if rare.size:
        log.warning(
            "%d classe(s) com < 2 exemplos excluída(s) da validação cruzada "
            "(o XGBoost exige todas as classes presentes em todo fold de treino, "
            "impossível garantir com < 2 exemplos): %s. Modelo final ainda treina "
            "com TODOS os dados.", rare.size, list(rare),
        )

    if strategy == "loo":
        from sklearn.model_selection import LeaveOneOut
        return mask, LeaveOneOut()

    from sklearn.model_selection import StratifiedKFold
    requested = int(params.get("cv_folds", 5))
    remaining_counts = counts[counts >= 2]
    min_class = int(remaining_counts.min()) if remaining_counts.size else requested
    cv_folds = max(2, min(requested, min_class))
    if cv_folds < requested:
        log.warning(
            "cv_folds reduzido de %d para %d -- classe menos frequente (entre as "
            "que sobraram) tem só %d exemplo(s)", requested, cv_folds, min_class,
        )
    random_state = int(params.get("random_seed", 42))
    return mask, StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


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
    demand_profiles: pd.DataFrame,
    policy_labels: pd.DataFrame,
    params: dict,
) -> Tuple[dict, pd.DataFrame]:
    """
    Treina um classificador multiclasse (XGBoost → LightGBM → RandomForest
    como fallbacks) que mapeia features operacionais + PERFIL → melhor política.

    2026-08-18: entrada mudou de `demand_features` para `demand_profiles`
    (que tem as mesmas colunas + `operational_profile`) -- o perfil entra
    como feature one-hot, a pedido do usuário ("AIPE é seleção de política
    por perfil, não faz sentido não usar perfil como entrada"). Este é o
    modelo de PRODUÇÃO (usado por `apply_policy_selector`); a comparação
    com/sem perfil, isolada, está em `evaluate_profile_feature_gain`.

    Retorna:
      policy_selector_model  — dict com 'model', 'label_encoder', 'feature_cols'
      policy_selector_metrics — DataFrame com acurácia, F1 por política etc.
    """
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    import warnings

    random_state: int = params.get("random_seed", 42)
    model_type: str   = params.get("model", "xgboost")

    # Junta features (+ perfil) + rótulos
    df = demand_profiles.merge(policy_labels[KEYS + ["best_policy"]], on=KEYS, how="inner")
    df = df.dropna(subset=FEATURE_COLS + ["best_policy"])
    df, profile_cols = _onehot_profile(df)

    # Remove features ausentes do dataset
    available_feats = [c for c in FEATURE_COLS if c in df.columns] + profile_cols
    X = df[available_feats].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(df["best_policy"])

    clf = _build_classifier(model_type, random_state, params)

    # Validação cruzada -- padrão LOO (ver _build_cv), não exige mínimo por
    # classe. `y[cv_mask]` sozinho (no modo stratified_kfold) deixa um
    # BURACO na numeração das classes (ex.: removendo a classe 1, sobra
    # 0,2,3...) -- o XGBoost exige rótulos contíguos 0..k-1 e falha com
    # "Invalid classes inferred". Re-densifica com um LabelEncoder novo só
    # pra CV (no-op quando mask é tudo True, como em LOO).
    cv_mask, cv = _build_cv(y, params)
    y_cv = LabelEncoder().fit_transform(y[cv_mask])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            clf, X[cv_mask], y_cv,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=True,
        )

    # Treina modelo final em TODO o dataset (inclui as classes raras
    # excluídas da CV -- o modelo aprende com elas, só não são avaliadas).
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
# Node 2b — Mede o ganho de usar o perfil como feature (pedido do usuário)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_profile_feature_gain(
    demand_profiles: pd.DataFrame,
    policy_labels: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Responde diretamente: "o ganho de usar perfil em relação a não usar é
    relevante?" (pedido do usuário). Treina DUAS versões do classificador
    -- COM e SEM `operational_profile` como feature -- usando o MESMO
    split de validação cruzada (mesma seed) pra comparação justa, e
    devolve a diferença de acurácia/F1.

    O perfil é categórico e DERIVADO dos mesmos FEATURE_COLS contínuos por
    um conjunto de regras de limiar (`classify_operational_profiles`) --
    em princípio um classificador flexível poderia reconstruir essas
    fronteiras sozinho a partir das features cruas. Esta função mede se
    isso de fato acontece nesta base, em vez de assumir.

    Retorna DataFrame de 1 linha com `cv_accuracy_sem_perfil`,
    `cv_accuracy_com_perfil`, `ganho_accuracy`, `ganho_accuracy_pct`,
    (idem para f1_macro) e `n_series`.
    """
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import LabelEncoder
    import warnings

    random_state: int = params.get("random_seed", 42)
    model_type: str   = params.get("model", "xgboost")

    df = demand_profiles.merge(policy_labels[KEYS + ["best_policy"]], on=KEYS, how="inner")
    df = df.dropna(subset=FEATURE_COLS + ["best_policy"])
    df, profile_cols = _onehot_profile(df)

    le = LabelEncoder()
    y = le.fit_transform(df["best_policy"])

    feats_sem = [c for c in FEATURE_COLS if c in df.columns]
    feats_com = feats_sem + profile_cols

    # MESMO cv (mesma seed) pras duas versões -- comparação pareada, não
    # dois experimentos independentes com splits diferentes. Padrão LOO
    # (ver _build_cv) -- não exige mínimo por classe.
    cv_mask, cv = _build_cv(y, params)
    y_cv = LabelEncoder().fit_transform(y[cv_mask])

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag, feats in [("sem_perfil", feats_sem), ("com_perfil", feats_com)]:
            X = df[feats].values.astype(float)
            clf = _build_classifier(model_type, random_state, params)
            cv_results = cross_validate(
                clf, X[cv_mask], y_cv, cv=cv,
                scoring=["accuracy", "f1_macro"],
            )
            results[tag] = {
                "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
                "accuracy_std":  float(np.std(cv_results["test_accuracy"])),
                "f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
            }

    acc_sem, acc_com = results["sem_perfil"]["accuracy_mean"], results["com_perfil"]["accuracy_mean"]
    f1_sem, f1_com = results["sem_perfil"]["f1_macro_mean"], results["com_perfil"]["f1_macro_mean"]
    ganho_acc = acc_com - acc_sem
    ganho_acc_pct = 100.0 * ganho_acc / acc_sem if acc_sem else float("nan")
    ganho_f1 = f1_com - f1_sem

    out = pd.DataFrame([{
        "n_series": int(len(df)),
        "cv_strategy": str(params.get("cv_strategy", "loo")),
        "n_folds": cv.get_n_splits(np.zeros((int(cv_mask.sum()), 1))) if hasattr(cv, "get_n_splits") else None,
        "cv_accuracy_sem_perfil":  round(acc_sem, 4),
        "cv_accuracy_com_perfil":  round(acc_com, 4),
        "ganho_accuracy":          round(ganho_acc, 4),
        "ganho_accuracy_pct":      round(ganho_acc_pct, 1),
        "cv_f1_macro_sem_perfil":  round(f1_sem, 4),
        "cv_f1_macro_com_perfil":  round(f1_com, 4),
        "ganho_f1_macro":          round(ganho_f1, 4),
        "relevante": bool(abs(ganho_acc) >= results["sem_perfil"]["accuracy_std"]),
    }])

    log.info(
        "Ganho de usar perfil como feature: accuracy %.3f -> %.3f (%+.1f%%) | "
        "F1_macro %.3f -> %.3f | %s",
        acc_sem, acc_com, ganho_acc_pct, f1_sem, f1_com,
        "RELEVANTE (>= 1 desvio-padrão do CV)" if out["relevante"].iloc[0] else "dentro do ruído do CV",
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Aplica o meta-modelo a todas as séries
# ─────────────────────────────────────────────────────────────────────────────

def apply_policy_selector(
    demand_profiles: pd.DataFrame,
    policy_selector_model: dict,
) -> pd.DataFrame:
    """
    Aplica o modelo treinado a demand_profiles e retorna a política recomendada
    com as probabilidades de cada classe.

    2026-08-18: entrada mudou de `demand_features` para `demand_profiles`
    (mesmas colunas + `operational_profile`) -- `feature_cols` do modelo
    agora inclui as colunas one-hot `profile_*` (ver `train_policy_selector`).

    Saída: DataFrame com KEYS + ['recommended_policy', 'confidence', prob_<policy>...]
    """
    clf    = policy_selector_model["model"]
    le     = policy_selector_model["label_encoder"]
    feats  = policy_selector_model["feature_cols"]

    demand_profiles, _ = _onehot_profile(demand_profiles)

    # Garante que todas as features necessárias estejam presentes
    missing = [f for f in feats if f not in demand_profiles.columns]
    if missing:
        log.warning("Features ausentes em demand_profiles: %s — imputadas com 0", missing)
        for m in missing:
            demand_profiles = demand_profiles.copy()
            demand_profiles[m] = 0.0

    X = demand_profiles[feats].fillna(0.0).values.astype(float)

    y_pred = clf.predict(X)
    recommended = le.inverse_transform(y_pred)

    result = demand_profiles[KEYS].copy()
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
