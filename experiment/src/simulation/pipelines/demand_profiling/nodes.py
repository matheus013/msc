"""
demand_profiling/nodes.py — Feature engineering operacional avançado.

Transforma séries temporais em vetores de features que caracterizam o regime
de demanda de cada (warehouse, store_id, item_id). Essas features alimentam
o Policy Selection Engine (pipeline policy_selection).

Duas saídas:
  demand_features   — DataFrame com ~20 features numéricas por série
  demand_profiles   — scenarios_meta enriquecido com perfil operacional e
                       política dominante recomendada (heurística)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

log = logging.getLogger(__name__)

KEYS = ["warehouse", "store_id", "item_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Features por série
# ─────────────────────────────────────────────────────────────────────────────

def compute_demand_features(
    scenarios: pd.DataFrame,
    scenarios_meta: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Calcula features operacionais por série (warehouse, store_id, item_id).

    Features geradas:
      Intermitência:
        adi              — Average Demand Interval = n_periods / n_positive
        cv2              — (sigma/mu)²
        intermittency_ratio — n_positive / n_periods
        zero_streak_max  — comprimento máximo de sequência de zeros
        zero_streak_mean — comprimento médio das sequências de zeros
        burstiness       — (std_inter - mean_inter) / (std_inter + mean_inter)
                           Goh & Barabasi (2008); [-1, 1]: -1=regular, +1=bursty

      Distribuição:
        demand_skewness  — assimetria da demanda (apenas pontos positivos)
        demand_kurtosis  — curtose excedente (apenas pontos positivos)
        entropy          — entropia de Shannon da distribuição normalizada
        p_zero           — fração de ciclos com demanda = 0

      Tendência e sazonalidade:
        trend_coef       — coeficiente OLS (unidades/ciclo) normalizado por mu
        seasonality_acf  — autocorrelação no lag sazonal (cycles_per_year)

      Volume:
        mu               — média da demanda
        sigma            — desvio padrão
        cv               — coeficiente de variação
        n_periods        — comprimento da série
        n_positive       — número de ciclos com demanda > 0
    """
    cycles_per_year: int = params.get("cycles_per_year", 17)

    records = []
    grouped = scenarios.sort_values(KEYS + ["venda_ciclo"]).groupby(KEYS)

    for key, grp in grouped:
        demand = grp["venda_ciclo"].sort_values()
        demand = grp.set_index("venda_ciclo")["demand"].sort_index().values.astype(float)

        feat = _series_features(demand, cycles_per_year)
        rec = dict(zip(KEYS, key if isinstance(key, tuple) else (key,)))
        rec.update(feat)
        records.append(rec)

    df_feat = pd.DataFrame(records)

    # Enriquece com colunas de perfil já presentes em scenarios_meta
    meta_cols = [c for c in scenarios_meta.columns
                 if c not in df_feat.columns or c in KEYS]
    df_feat = df_feat.merge(
        scenarios_meta[meta_cols],
        on=KEYS,
        how="left",
    )

    log.info(
        "demand_features: %d séries × %d features",
        len(df_feat),
        df_feat.shape[1] - len(KEYS),
    )
    return df_feat.reset_index(drop=True)


def _series_features(demand: np.ndarray, cycles_per_year: int) -> Dict[str, float]:
    """Calcula todas as features para uma única série de demanda."""
    n = len(demand)
    n_positive = int((demand > 0).sum())
    mu = float(demand.mean())
    sigma = float(demand.std(ddof=1)) if n > 1 else 0.0
    cv = sigma / (mu + 1e-9)

    feat: Dict[str, float] = {
        "mu": mu,
        "sigma": sigma,
        "cv": cv,
        "cv2": cv ** 2,
        "n_periods": n,
        "n_positive": n_positive,
        "p_zero": 1.0 - n_positive / max(n, 1),
        "intermittency_ratio": n_positive / max(n, 1),
    }

    # ADI — Average Demand Interval
    feat["adi"] = n / max(n_positive, 1)

    # Zero streaks
    zero_runs = _zero_run_lengths(demand)
    if zero_runs:
        feat["zero_streak_max"] = float(max(zero_runs))
        feat["zero_streak_mean"] = float(np.mean(zero_runs))
    else:
        feat["zero_streak_max"] = 0.0
        feat["zero_streak_mean"] = 0.0

    # Burstiness (Goh & Barabasi 2008) — baseada nos intervalos entre demandas
    inter_event = _inter_event_times(demand)
    if len(inter_event) >= 2:
        m_i = float(np.mean(inter_event))
        s_i = float(np.std(inter_event, ddof=1))
        feat["burstiness"] = (s_i - m_i) / (s_i + m_i + 1e-9)
    else:
        feat["burstiness"] = 0.0

    # Distribuição dos valores positivos
    pos_vals = demand[demand > 0]
    if len(pos_vals) >= 3:
        feat["demand_skewness"] = float(_skewness(pos_vals))
        feat["demand_kurtosis"] = float(_kurtosis(pos_vals))
    else:
        feat["demand_skewness"] = 0.0
        feat["demand_kurtosis"] = 0.0

    # Entropia de Shannon (demanda normalizada, bins=10)
    feat["entropy"] = float(_shannon_entropy(demand, bins=10))

    # Tendência linear (coeficiente OLS / mu)
    feat["trend_coef"] = float(_linear_trend(demand)) / (mu + 1e-9)

    # Sazonalidade: autocorrelação no lag sazonal
    lag = cycles_per_year
    if n > lag + 1:
        feat["seasonality_acf"] = float(_autocorr(demand, lag))
    else:
        feat["seasonality_acf"] = 0.0

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 2. Perfis operacionais de demanda
# ─────────────────────────────────────────────────────────────────────────────

def classify_operational_profiles(
    demand_features: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Classifica cada série em um dos 5 Perfis Operacionais de Demanda (POD)
    e atribui a política dominante recomendada (heurística baseada em regras).

    Perfis (em ordem de prioridade):
      Sparse_High_Impact  — ADI alto + CV² alto + streak longo → GA-DQN
      High_Vol_Seasonal   — sazonalidade forte ou burstiness alto  → PPO
      Unstable_Trend      — tendência acentuada ou alta entropia   → GA-PPO
      Low_Vol_Stable      — baixo volume, baixo CV²                → Newsvendor
      Fast_Moving         — ADI baixo                              → sS

    A coluna `dominant_policy` é o rótulo-alvo do Policy Selector (pipeline B).
    """
    thresholds = params.get("profile_thresholds", {})
    adi_high       = thresholds.get("adi_high", 1.32)
    cv2_high       = thresholds.get("cv2_high", 0.49)
    streak_high    = thresholds.get("zero_streak_max_high", 5.0)
    burst_high     = thresholds.get("burstiness_high", 0.3)
    acf_high       = thresholds.get("seasonality_acf_high", 0.25)
    trend_high     = thresholds.get("trend_coef_high", 0.05)
    entropy_high   = thresholds.get("entropy_high", 2.0)
    mu_low         = thresholds.get("mu_low", 2.0)

    df = demand_features.copy()

    conditions = [
        # 1. Sparse High Impact — lumpy com longos silêncios → GA-DQN
        (df["adi"] >= adi_high) & (df["cv2"] >= cv2_high) & (df["zero_streak_max"] >= streak_high),
        # 2. High Volatility Seasonal — sazonalidade ou burstiness forte → PPO
        (df["seasonality_acf"] >= acf_high) | (df["burstiness"] >= burst_high),
        # 3. Unstable Trend — tendência ou alta entropia → GA-PPO
        (df["trend_coef"].abs() >= trend_high) | (df["entropy"] >= entropy_high),
        # 4. Low Volume Stable — baixo volume, pouca variação → Newsvendor
        (df["mu"] <= mu_low) & (df["cv2"] < cv2_high),
        # 5. Fast Moving — demanda frequente → (s,S)
        df["adi"] < adi_high,
    ]
    choices = [
        "Sparse_High_Impact",
        "High_Vol_Seasonal",
        "Unstable_Trend",
        "Low_Vol_Stable",
        "Fast_Moving",
    ]
    policy_map = {
        "Sparse_High_Impact": "GA-DQN",
        "High_Vol_Seasonal":  "PPO",
        "Unstable_Trend":     "GA-PPO",
        "Low_Vol_Stable":     "Newsvendor",
        "Fast_Moving":        "sS",
    }

    df["operational_profile"] = np.select(conditions, choices, default="Sparse_High_Impact")
    df["dominant_policy"]     = df["operational_profile"].map(policy_map)

    dist = df["operational_profile"].value_counts().to_dict()
    log.info("Perfis operacionais: %s", dist)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers matemáticos (sem dependências externas além de numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _zero_run_lengths(demand: np.ndarray):
    """Retorna lista com comprimentos de cada sequência consecutiva de zeros."""
    runs = []
    count = 0
    for d in demand:
        if d == 0:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def _inter_event_times(demand: np.ndarray):
    """Intervalos (em ciclos) entre demandas positivas consecutivas."""
    idx = np.where(demand > 0)[0]
    if len(idx) < 2:
        return np.array([])
    return np.diff(idx).astype(float)


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-9:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-9:
        return 0.0
    return float(((x - m) ** 4).mean() / s ** 4) - 3.0  # excesso


def _shannon_entropy(demand: np.ndarray, bins: int = 10) -> float:
    counts, _ = np.histogram(demand, bins=bins)
    counts = counts[counts > 0].astype(float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _linear_trend(demand: np.ndarray) -> float:
    """Coeficiente angular da regressão OLS simples sobre o índice temporal."""
    n = len(demand)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean()
    ym = demand.mean()
    num = ((x - xm) * (demand - ym)).sum()
    den = ((x - xm) ** 2).sum()
    return float(num / (den + 1e-9))


def _autocorr(demand: np.ndarray, lag: int) -> float:
    """Autocorrelação da série no lag especificado."""
    n = len(demand)
    if n <= lag:
        return 0.0
    m = demand.mean()
    var = ((demand - m) ** 2).mean()
    if var < 1e-9:
        return 0.0
    cov = ((demand[:n - lag] - m) * (demand[lag:] - m)).mean()
    return float(cov / var)
