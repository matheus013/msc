"""
data_loader.py — Carregamento e pré-processamento dos dados
"""
import pandas as pd
import numpy as np
from pathlib import Path


def _parse_venda_ciclo(series: pd.Series, cycles_per_year: int = 17) -> pd.Series:
    """Converte venda_ciclo (inteiro YYYYCC) para datetime. Ex: 202313 -> ~2023-09-01"""
    v = series.astype(int)
    year = v // 100
    part = (v % 100) - 1  # 0-indexed
    days_per_cycle = int(365 / cycles_per_year)
    base = pd.to_datetime({"year": year, "month": 1, "day": 1})
    return base + pd.to_timedelta(part * days_per_cycle, unit="D")


def _parse_date_column(series: pd.Series, cycles_per_year: int = 17) -> pd.Series:
    """
    Parseia coluna de data que pode estar no formato ciclo (YYYY-CC-01, CC=01-17)
    ou no formato inteiro YYYYCC.
    """
    if pd.api.types.is_numeric_dtype(series):
        return _parse_venda_ciclo(series, cycles_per_year)

    dates = pd.to_datetime(series, errors="coerce")
    if dates.isna().mean() > 0.05:
        # Muitas datas inválidas — provavelmente formato YYYY-CC-DD com ciclo > 12
        try:
            parts = series.str.extract(r"(\d{4})-(\d{2})-\d{2}", expand=True)
            synthetic = parts[0].astype(int) * 100 + parts[1].astype(int)
            dates = _parse_venda_ciclo(synthetic, cycles_per_year)
        except Exception:
            pass
    return dates


def load_data(cfg: dict) -> pd.DataFrame:
    """Carrega e normaliza o DataFrame conforme configuração."""
    dcfg = cfg["DATA"]
    path = Path(dcfg["file_path"])
    fmt = dcfg.get("file_format", "csv").lower()

    if fmt == "csv":
        df = pd.read_csv(path, sep=dcfg.get("csv_sep", ","))
    elif fmt == "excel":
        df = pd.read_excel(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Formato não suportado: {fmt}")

    cols = dcfg["columns"]

    rename_map = {}
    for internal, user_col in cols.items():
        if user_col and user_col in df.columns:
            rename_map[user_col] = internal
    df = df.rename(columns=rename_map)

    if "demand" not in df.columns:
        raise ValueError(
            f"Coluna de demanda '{cols['demand']}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    # Remove sentinelas INT64 (≈ -9.2e18 de sistemas BigQuery/Spark), negativos e outliers
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    valid = df["demand"].notna() & (df["demand"] >= 0) & (df["demand"] < 100_000)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"[DataLoader] Removidas {n_dropped} linhas com demand inválido/extremo "
              f"({n_dropped / len(df):.1%} do total)")
        df = df[valid]

    # Prefere venda_ciclo (inteiro YYYYCC) sobre a coluna date (que pode ter meses inválidos 13-17)
    if "venda_ciclo" in df.columns:
        df["date"] = _parse_venda_ciclo(df["venda_ciclo"])
    elif "date" in df.columns:
        df["date"] = _parse_date_column(df["date"])

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Pré-filtro por loja antes da agregação (se especificado)
    if "store_id" in df.columns and dcfg.get("selected_store") is not None:
        sel_store = dcfg["selected_store"]
        df = df[df["store_id"] == sel_store]

    # Converte dados transacionais (item+loja+ciclo) para demanda total por período
    group_cols = [c for c in ["date", "item_id", "warehouse"] if c in df.columns]
    if len(group_cols) >= 2:
        agg = {"demand": "sum"}
        for col, method in {"revenue": "sum", "avg_price": "mean"}.items():
            if col in df.columns:
                agg[col] = method
        df = df.groupby(group_cols, sort=False).agg(agg).reset_index()
        print(f"[DataLoader] Agregados para {len(df):,} registros "
              f"(item×ciclo×warehouse) | "
              f"Demanda mediana por período: {df['demand'].median():.1f}")

    if dcfg.get("filter_single_item", True):
        df = _filter_single_series(df, dcfg)

    df = df.dropna(subset=["demand"])
    df["demand"] = df["demand"].clip(lower=0).astype(float)

    print(f"[DataLoader] Dados carregados: {len(df)} períodos | "
          f"Demanda média: {df['demand'].mean():.2f} | "
          f"Std: {df['demand'].std():.2f} | "
          f"Ciclos/ano: 17")

    return df


def _filter_single_series(df: pd.DataFrame, dcfg: dict) -> pd.DataFrame:
    """
    Seleciona a série temporal de um único warehouse/item.
    Após a agregação, cada linha já representa um único período (ciclo) por item+warehouse.
    """
    cols = dcfg["columns"]
    has_warehouse = "warehouse" in df.columns and cols.get("warehouse")
    has_item = "item_id" in df.columns and cols.get("item_id")

    if has_warehouse:
        sel_warehouse = dcfg.get("selected_warehouse")
        if sel_warehouse is None:
            sel_warehouse = df.groupby("warehouse")["demand"].sum().idxmax()
            print(f"[DataLoader] Selecionado warehouse: {sel_warehouse}")
        df = df[df["warehouse"] == sel_warehouse]

    if has_item:
        sel_item = dcfg.get("selected_item")
        if sel_item is None:
            sel_item = df.groupby("item_id").size().idxmax()
            print(f"[DataLoader] Selecionado item: {sel_item}")
        df = df[df["item_id"] == sel_item]

    return df.reset_index(drop=True)


def train_test_split(df: pd.DataFrame, cfg: dict):
    """Divide em treino e teste."""
    dcfg = cfg["DATA"]
    demand = df["demand"].values

    test_start = dcfg.get("test_start_date")
    if test_start and "date" in df.columns:
        split_idx = (df["date"] >= pd.to_datetime(test_start)).idxmax()
    else:
        split_idx = int(len(demand) * dcfg.get("train_ratio", 0.8))

    train = demand[:split_idx]
    test = demand[split_idx:]
    print(f"[DataLoader] Treino: {len(train)} períodos | Teste: {len(test)} períodos")
    return train, test


def build_features(demand: np.ndarray, lookback: int,
                   extra_cols: np.ndarray = None):
    """Cria janelas deslizantes (X, y) para modelos supervisionados."""
    n = len(demand) - lookback
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    X = demand[idx]
    y = demand[lookback:]
    if extra_cols is not None:
        X = np.hstack([X, extra_cols[lookback:]])
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# FUNÇÕES PARA ANÁLISE MULTI-NÍVEL (Rede de Suprimento)
# ═══════════════════════════════════════════════════════════════════════════

def get_network_structure(df: pd.DataFrame) -> dict:
    """Analisa e retorna a estrutura da rede de suprimento."""
    structure = {}

    has_warehouse = "warehouse" in df.columns
    has_store = "store_id" in df.columns
    has_item = "item_id" in df.columns

    if has_warehouse:
        warehouses = df["warehouse"].unique()
        structure["warehouses"] = sorted(warehouses.astype(str))

        if has_store:
            structure["stores_per_warehouse"] = {
                wh: sorted(df[df["warehouse"] == wh]["store_id"].unique().astype(str))
                for wh in warehouses
            }

    if has_item:
        structure["items"] = sorted(df["item_id"].unique().astype(str))

    if "date" in df.columns:
        structure["periods"] = len(df["date"].unique())

    if has_warehouse and has_item and "date" in df.columns:
        n_wh = len(df["warehouse"].unique())
        n_items = len(df["item_id"].unique())
        n_periods = len(df["date"].unique())
        structure["density"] = f"{100 * len(df) / (n_wh * n_items * n_periods):.1f}%"

    return structure


def aggregate_warehouse_demand(df: pd.DataFrame, warehouse: str) -> np.ndarray:
    """Agrega demanda de todos os itens dentro de um warehouse por período."""
    wh_data = df[df["warehouse"] == warehouse]
    if "date" in wh_data.columns:
        return wh_data.groupby("date")["demand"].sum().values.astype(float)
    return wh_data["demand"].values.astype(float)


def get_warehouse_context(df: pd.DataFrame, warehouse: str) -> dict:
    """Retorna informações de contexto de um warehouse específico."""
    wh_data = df[df["warehouse"] == warehouse]
    return {
        "warehouse": warehouse,
        "n_stores": wh_data["store_id"].nunique() if "store_id" in wh_data.columns else 1,
        "n_products": wh_data["item_id"].nunique() if "item_id" in wh_data.columns else 1,
        "periods": len(wh_data),
        "total_demand": wh_data["demand"].sum(),
        "avg_demand": wh_data["demand"].mean(),
        "std_demand": wh_data["demand"].std(),
    }
