"""
data_ingestion/nodes.py — Carregamento, filtro, limpeza e construção de cenários.

Colunas reais no dataset:
  produto_cod, revendedor_cod, venda_data, venda_qtd,
  venda_vlr_receita_liquida, venda_vlr_receita_bruta,
  venda_tipo, venda_ciclo, id, estrutura, genero, idade,
  segmento, status, filial, praca, gerente_regional, ciclos_inativos

Regras de filtro aplicadas aqui (configuráveis via YAML):
  • Lojas:   status IN active_statuses (default ["Ativo"])
  • Produtos: ativos = vistos nos últimos active_product_window ciclos
  • NÃO filtrar por volume de vendas — o campo segmento já stratifica por receita
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

log = logging.getLogger(__name__)

# Colunas de perfil da revendedora (constantes por revendedor_cod)
STORE_PROFILE_COLS = [
    "segmento", "genero", "idade", "estrutura",
    "filial", "praca", "gerente_regional", "ciclos_inativos",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carregamento
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_sales(partitioned_input: Dict[str, Any], params: dict) -> pd.DataFrame:
    """
    Carrega os parquet particionados por estado (uf=XX/).
    Suporta states=["all"] ou states=["PB","SP",...].
    """
    states = params.get("states", ["all"])
    load_all = (states == ["all"] or states == "all")

    frames = []
    for partition_key, load_fn in partitioned_input.items():
        uf = _extract_uf(partition_key)
        if not load_all and uf not in [s.upper() for s in states]:
            continue
        df = load_fn()
        df["_uf"] = uf
        frames.append(df)
        log.info("Carregado: %s (%d linhas)", partition_key, len(df))

    if not frames:
        raise ValueError(f"Nenhuma partição encontrada para states={states}")

    raw = pd.concat(frames, ignore_index=True)
    log.info("Total carregado: %d linhas, %d colunas | colunas: %s",
             len(raw), raw.shape[1], raw.columns.tolist())
    return raw


def _extract_uf(partition_key: str) -> str:
    key = partition_key.replace("\\", "/").split("/")[0]
    if "=" in key:
        return key.split("=")[1].upper()
    return key.upper()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normalização e filtros
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia nomes de colunas existentes para os nomes canônicos do projeto.
    Colunas de perfil (segmento, genero, …) são preservadas com o nome original.
    """
    col_map = {
        # warehouse / estado
        "_uf":          "warehouse",
        "uf":           "warehouse",
        "armazem":      "warehouse",
        "estado":       "warehouse",
        "cd_uf":        "warehouse",
        # store / revendedora
        "revendedor_cod": "store_id",
        "revendedor":   "store_id",
        "loja":         "store_id",
        "store":        "store_id",
        "cd_loja":      "store_id",
        "pdv":          "store_id",
        # item / produto
        "produto_cod":  "item_id",
        "produto":      "item_id",
        "item":         "item_id",
        "sku":          "item_id",
        "cd_produto":   "item_id",
        "cod_produto":  "item_id",
        # ciclo (já está como venda_ciclo no dataset — mapeado por precaução)
        "ciclo":        "venda_ciclo",
        "periodo":      "venda_ciclo",
        "dt_ciclo":     "venda_ciclo",
        "bimestre":     "venda_ciclo",
        # demanda (quantidade vendida)
        "venda_qtd":    "demand",
        "quantidade":   "demand",
        "vendas":       "demand",
        "qtd":          "demand",
        "qtd_vendida":  "demand",
        "sales":        "demand",
        "venda":        "demand",
        # receita
        "venda_vlr_receita_liquida": "revenue",
        "venda_vlr_receita_bruta":   "revenue_gross",
    }
    rename = {c: col_map[c.lower()] for c in df.columns if c.lower() in col_map}
    return df.rename(columns=rename)


def filter_by_parameters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Aplica os filtros de seleção configuráveis:
      1. Normaliza nomes de colunas
      2. Filtra lojas pelo campo 'status' (mantém apenas active_statuses)
      3. Filtra produtos ativos (vistos nos últimos active_product_window ciclos)
      4. Filtra por produto específico (se 'products' configurado)
      5. Filtra por período de interesse (date_start / date_end)
      6. Filtra por segmento (opcional)
      7. Filtra por CV mínimo (demanda suficientemente variável)
    """
    df = df.copy()
    df = _normalize_columns(df)

    # ── 1. Garante tipos corretos antes de filtrar ─────────────────────────
    if "demand" in df.columns:
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0)
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "revenue_gross" in df.columns:
        df["revenue_gross"] = pd.to_numeric(df["revenue_gross"], errors="coerce").fillna(0.0)

    initial = len(df)

    # ── 2. Filtro por status da loja ───────────────────────────────────────
    active_statuses = params.get("active_statuses", ["Ativo"])
    if "status" in df.columns and active_statuses:
        before = len(df)
        df = df[df["status"].isin(active_statuses)]
        log.info("Filtro status %s: %d -> %d linhas (%d lojas removidas)",
                 active_statuses, before, len(df), before - len(df))

    # ── 2b. Filtro por tipo de venda (remove brindes/amostras) ────────────
    exclude_tipos = params.get("exclude_venda_tipos")
    if exclude_tipos and "venda_tipo" in df.columns:
        before = len(df)
        df = df[~df["venda_tipo"].isin(exclude_tipos)]
        log.info("Filtro venda_tipo (excluindo %s): %d -> %d linhas (-%d)",
                 exclude_tipos, before, len(df), before - len(df))
    elif exclude_tipos and "venda_tipo" not in df.columns:
        log.warning("exclude_venda_tipos configurado mas coluna 'venda_tipo' nao encontrada")

    # ── 3. Filtro por produtos ativos (vistos em ciclos recentes) ──────────
    active_product_window = params.get("active_product_window", 3)
    if "item_id" in df.columns and "venda_ciclo" in df.columns and active_product_window > 0:
        sorted_ciclos = sorted(df["venda_ciclo"].unique(), reverse=True)
        recent_ciclos = sorted_ciclos[:active_product_window]
        active_products = df[df["venda_ciclo"].isin(recent_ciclos)]["item_id"].unique()
        before = len(df)
        df = df[df["item_id"].isin(active_products)]
        log.info("Filtro produtos ativos (janela=%d ciclos): %d produtos ativos | %d -> %d linhas",
                 active_product_window, len(active_products), before, len(df))

    # ── 4. Filtro por produto específico ──────────────────────────────────
    products = params.get("products")
    if products is not None and len(products) > 0:
        df["item_id"] = df["item_id"].astype(str)
        df = df[df["item_id"].isin([str(p) for p in products])]
        log.info("Filtro produto %s: %d linhas", products, len(df))

    # ── 5. Filtro por período ──────────────────────────────────────────────
    date_start = params.get("date_start")
    date_end   = params.get("date_end")
    if date_start:
        df = df[df["venda_ciclo"].astype(str) >= str(date_start)]
    if date_end:
        df = df[df["venda_ciclo"].astype(str) <= str(date_end)]
    log.info("Após filtro de período [%s, %s]: %d linhas", date_start, date_end, len(df))

    # ── 6. Filtro por segmento (opcional) ─────────────────────────────────
    segmentos = params.get("segmentos")
    if segmentos and "segmento" in df.columns:
        before = len(df)
        df = df[df["segmento"].isin(segmentos)]
        log.info("Filtro segmento %s: %d -> %d linhas", segmentos, before, len(df))

    # ── 7. Filtro por CV mínimo (pulado quando produtos específicos já escolhidos) ──
    products = params.get("products")
    cv_threshold = params.get("cv_threshold", 0.0)
    if cv_threshold > 0 and "item_id" in df.columns and not products:
        cv_per_item = (df.groupby("item_id")["demand"]
                       .agg(lambda x: x.std() / (x.mean() + 1e-9)))
        valid_items = cv_per_item[cv_per_item >= cv_threshold].index
        before = len(df)
        df = df[df["item_id"].isin(valid_items)]
        log.info("Filtro CV >= %.2f: %d -> %d linhas, %d produtos",
                 cv_threshold, before, len(df), df["item_id"].nunique())

    log.info("filter_by_parameters: %d -> %d linhas (-%d)",
             initial, len(df), initial - len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Limpeza
# ─────────────────────────────────────────────────────────────────────────────

def clean_sales_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Limpeza de dados:
      - Deduplicação
      - Remove demanda negativa
      - Preenche ciclos ausentes
      - Remove lojas sem nenhuma venda no horizonte

    NÃO filtra por volume máximo de vendas — o campo 'segmento' já
    estratifica por faturamento; remover top-sellers distorceria a análise.
    """
    df = df.copy()
    initial_len = len(df)

    # Deduplicação
    dedup_cols = params.get("dedup_by",
                             ["warehouse", "item_id", "store_id", "venda_ciclo"])
    dedup_present = [c for c in dedup_cols if c in df.columns]
    if dedup_present:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_present)
        log.info("Deduplicação: removidas %d linhas duplicadas", before - len(df))

    # Remove demanda negativa
    if params.get("remove_negative_sales", True) and "demand" in df.columns:
        mask = df["demand"] >= 0
        removed = (~mask).sum()
        if removed > 0:
            df = df[mask]
            log.info("Removidas %d linhas com demanda negativa", removed)

    # Remove lojas sem nenhuma venda no horizonte completo
    if params.get("remove_zero_only_stores", True):
        store_sum = df.groupby(["warehouse", "store_id", "item_id"])["demand"].sum()
        valid = store_sum[store_sum > 0].reset_index()[["warehouse", "store_id", "item_id"]]
        before = len(df)
        df = df.merge(valid, on=["warehouse", "store_id", "item_id"], how="inner")
        log.info("Lojas sem venda alguma: removidas %d linhas", before - len(df))

    log.info("Limpeza: %d -> %d linhas (-%d)",
             initial_len, len(df), initial_len - len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Construção dos cenários
# ─────────────────────────────────────────────────────────────────────────────

def build_demand_scenarios(df: pd.DataFrame, params: dict) -> tuple:
    """
    Constrói séries temporais por (warehouse, store_id, item_id).

    scenarios:  DataFrame long com colunas
        [warehouse, store_id, item_id, venda_ciclo, demand, revenue?, revenue_gross?]

    scenarios_meta: DataFrame com estatísticas e perfil por série:
        mu, sigma, cv, n_periods, n_positive, group,
        mu_revenue, sigma_revenue,           — se revenue disponível
        segmento, genero, idade, estrutura,   — perfil da revendedora
        filial, praca, gerente_regional,
        ci_status, ci_ciclos                  — atividade recente
    """
    min_positive_cycles = params.get("min_positive_cycles", 17)
    max_stores          = params.get("max_stores")
    cycles_per_year     = params.get("cycles_per_year", 17)

    # ── Extrai perfil da revendedora (constante por store_id) ─────────────
    present_profile = [c for c in STORE_PROFILE_COLS if c in df.columns]
    store_profile = pd.DataFrame()
    if present_profile:
        def _mode_safe(x):
            m = x.dropna().mode()
            return m.iloc[0] if len(m) > 0 else None

        store_profile = (
            df.groupby(["warehouse", "store_id"])[present_profile]
            .agg(_mode_safe)
            .reset_index()
        )
        # Decodifica ciclos_inativos: "A0","A1" -> ativo; "I4","I9" -> inativo
        if "ciclos_inativos" in store_profile.columns:
            ci = store_profile["ciclos_inativos"].fillna("-")
            store_profile["ci_status"] = ci.str[:1]          # 'A', 'I' ou '-'
            store_profile["ci_ciclos"] = pd.to_numeric(
                ci.str[1:], errors="coerce")                  # número de ciclos
        log.info("Perfil extraído para %d revendedoras | colunas: %s",
                 len(store_profile), present_profile)

    # ── Agrega por ciclo ──────────────────────────────────────────────────
    has_revenue = "revenue" in df.columns
    has_revenue_gross = "revenue_gross" in df.columns
    agg_dict: dict = {"demand": "sum"}
    if has_revenue:
        agg_dict["revenue"] = "sum"
    if has_revenue_gross:
        agg_dict["revenue_gross"] = "sum"

    df_agg = (
        df.groupby(["warehouse", "store_id", "item_id", "venda_ciclo"])
        .agg(agg_dict)
        .reset_index()
    )

    all_cycles = sorted(df_agg["venda_ciclo"].unique())

    # Zero-fill: toda combinação (w, s, i) existente deve ter TODOS os ciclos.
    # Processado warehouse a warehouse para evitar cross-join de centenas de
    # milhões de linhas em memória de uma vez.
    if params.get("fill_missing_cycles", True):
        fill_vals = {"demand": 0}
        if has_revenue:
            fill_vals["revenue"] = 0.0
        if has_revenue_gross:
            fill_vals["revenue_gross"] = 0.0

        n_keys   = df_agg[["warehouse", "store_id", "item_id"]].drop_duplicates().__len__()
        n_cycles = len(all_cycles)
        log.info("Zero-fill: %d séries × %d ciclos = %d linhas (por warehouse)", n_keys, n_cycles, n_keys * n_cycles)

        cycles_df  = pd.DataFrame({"venda_ciclo": all_cycles})
        warehouses = df_agg["warehouse"].unique()
        chunks: list[pd.DataFrame] = []

        for wh in warehouses:
            wh_data = df_agg[df_agg["warehouse"] == wh]
            wh_keys = wh_data[["warehouse", "store_id", "item_id"]].drop_duplicates()
            wh_grid = wh_keys.merge(cycles_df, how="cross")
            wh_filled = wh_grid.merge(
                wh_data, on=["warehouse", "store_id", "item_id", "venda_ciclo"], how="left"
            )
            for col, val in fill_vals.items():
                if col in wh_filled.columns:
                    wh_filled[col] = wh_filled[col].fillna(val)
            chunks.append(wh_filled)

        df_agg = pd.concat(chunks, ignore_index=True)

    # Filtra séries com ciclos positivos insuficientes
    n_positive = (df_agg.groupby(["warehouse", "store_id", "item_id"])["demand"]
                  .apply(lambda x: (x > 0).sum())
                  .reset_index(name="_n_pos"))
    valid_keys = n_positive[n_positive["_n_pos"] >= min_positive_cycles][
        ["warehouse", "store_id", "item_id"]]
    df_agg = df_agg.merge(valid_keys, on=["warehouse", "store_id", "item_id"], how="inner")

    # ── Calcula metadados por série (vetorizado) ──────────────────────────
    keys = ["warehouse", "store_id", "item_id"]
    grp_d = df_agg.groupby(keys)["demand"]

    scenarios_meta = grp_d.agg(
        mu="mean",
        sigma="std",
        n_periods="count",
        n_positive=lambda x: (x > 0).sum(),
    ).reset_index()
    scenarios_meta["sigma"] = scenarios_meta["sigma"].fillna(0.0)
    scenarios_meta["cv"]    = scenarios_meta["sigma"] / (scenarios_meta["mu"] + 1e-9)

    if has_revenue and "revenue" in df_agg.columns:
        grp_r = df_agg.groupby(keys)["revenue"].agg(
            mu_revenue="mean", sigma_revenue="std"
        ).reset_index()
        grp_r["sigma_revenue"] = grp_r["sigma_revenue"].fillna(0.0)
        grp_r["cv_revenue"]    = grp_r["sigma_revenue"] / (grp_r["mu_revenue"] + 1e-9)
        scenarios_meta = scenarios_meta.merge(grp_r, on=keys, how="left")

    # Syntetos-Boylan vetorizado: evita apply linha-a-linha
    adi  = scenarios_meta["n_periods"] / scenarios_meta["n_positive"].clip(lower=1)
    cv2  = scenarios_meta["cv"] ** 2
    scenarios_meta["group"] = np.where(
        (adi >= 1.32) & (cv2 >= 0.49), "Lumpy",
        np.where(
            adi >= 1.32, "Intermittent",
            np.where(cv2 >= 0.49, "Erratic", "Smooth")
        )
    )

    # Merge do perfil da revendedora
    if not store_profile.empty:
        scenarios_meta = scenarios_meta.merge(
            store_profile, on=["warehouse", "store_id"], how="left"
        )

    # Limita número de lojas por estado (opcional)
    if max_stores is not None:
        keep_frames = []
        for w in scenarios_meta["warehouse"].unique():
            sub = scenarios_meta[scenarios_meta["warehouse"] == w]
            keep_frames.append(
                sub.nlargest(max_stores, "n_positive")[["warehouse", "store_id", "item_id"]]
            )
        keep_keys = pd.concat(keep_frames, ignore_index=True)
        df_agg         = df_agg.merge(keep_keys,        on=["warehouse", "store_id", "item_id"], how="inner")
        scenarios_meta = scenarios_meta.merge(keep_keys, on=["warehouse", "store_id", "item_id"], how="inner")

    n_series = len(scenarios_meta)
    if n_series == 0:
        raise RuntimeError(
            "Nenhum cenario gerado apos filtros. "
            "Verifique: min_positive_cycles, cv_threshold, active_statuses, products."
        )
    log.info(
        "Cenarios: %d series | %d ciclos | grupos: %s | segmentos: %s",
        n_series, len(all_cycles),
        scenarios_meta["group"].value_counts().to_dict() if "group" in scenarios_meta.columns else "n/a",
        scenarios_meta["segmento"].value_counts().to_dict()
        if "segmento" in scenarios_meta.columns else "n/a",
    )

    scenarios = df_agg.sort_values(["warehouse", "store_id", "item_id", "venda_ciclo"])
    return scenarios.reset_index(drop=True), scenarios_meta.reset_index(drop=True)


def _classify_demand_group(cv: float, n_positive: int, n_total: int,
                            cycles_per_year: int) -> str:
    """
    Syntetos-Boylan 4-quadrant classification (Syntetos, Boylan & Croston 2005):
      ADI = n_total / n_positive  (average inter-demand interval)
      CV2 = cv^2                  (squared coefficient of variation of non-zero sizes)
      Thresholds: ADI = 1.32, CV2 = 0.49

      Smooth       : ADI <  1.32 and CV2 <  0.49
      Erratic      : ADI <  1.32 and CV2 >= 0.49
      Intermittent : ADI >= 1.32 and CV2 <  0.49
      Lumpy        : ADI >= 1.32 and CV2 >= 0.49
    """
    adi = n_total / max(n_positive, 1)
    cv2 = cv ** 2
    if adi >= 1.32 and cv2 >= 0.49:
        return "Lumpy"
    elif adi >= 1.32:
        return "Intermittent"
    elif cv2 >= 0.49:
        return "Erratic"
    else:
        return "Smooth"
