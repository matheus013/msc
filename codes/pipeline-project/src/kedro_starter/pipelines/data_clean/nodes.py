from __future__ import annotations

from typing import Optional

import pandas as pd


def filter_active_products(
        sales: pd.DataFrame,
        products: pd.DataFrame,
        sample_frac: Optional[float] = None,
        random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    # --- Coerce to DataFrame if a Series slipped in ---
    if isinstance(sales, pd.Series):
        sales = sales.to_frame()
    if isinstance(products, pd.Series):
        products = products.to_frame()

    # --- Basic type & column checks ---
    if not isinstance(sales, pd.DataFrame):
        raise TypeError(f"'sales' must be a DataFrame, got {type(sales)}")
    if not isinstance(products, pd.DataFrame):
        raise TypeError(f"'products' must be a DataFrame, got {type(products)}")

    required_cols_sales = {"produto_cod"}
    required_cols_products = {"produto_cod", "produto_status"}
    missing_sales = required_cols_sales - set(sales.columns)
    missing_products = required_cols_products - set(products.columns)
    if missing_sales:
        raise KeyError(f"Missing columns in sales: {sorted(missing_sales)}")
    if missing_products:
        raise KeyError(f"Missing columns in products: {sorted(missing_products)}")

    # --- somente vendas

    sales = sales[sales['venda_tipo'] == 'Venda']

    # --- Optional sampling (keeps DataFrame type) ---
    if sample_frac is not None:
        if not (0 < sample_frac <= 1):
            raise ValueError("sample_frac must be in (0, 1].")
        sales = sales.sample(frac=sample_frac, random_state=random_seed)

    # --- Normalize dtypes/values used for the join & filter ---
    sales = sales.copy()
    sales["produto_cod"] = sales["produto_cod"].astype("string").str.strip()

    products = products.copy()
    products["produto_cod"] = products["produto_cod"].astype("string").str.strip()
    products["produto_status"] = (
        products["produto_status"].astype("string").str.strip().str.lower()
    )

    # Keep only active products
    active_products = products.loc[
        products["produto_status"].eq("ativo"), ["produto_cod", "produto_status"]
    ]

    # --- Merge as DataFrames ---
    filtered_sales = sales.merge(active_products, on="produto_cod", how="inner")
    return filtered_sales.reset_index(drop=True)


def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # normalize columns you rely on downstream
    if "produto_status" in df.columns:
        df["produto_status"] = df["produto_status"].astype(str).str.strip().str.lower()
    if "produto_cod" in df.columns:
        df["produto_cod"] = df["produto_cod"].astype(str).str.strip()
    return df


def clean_revendedor(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    renames = {
        "revendedor_cod": "rev_id",
        "revendedor_bloqueio_cadastro": "cad_blocked",
        "revendedor_estrutura_comercial": "estrut_com",
        "revendedor_genero": "gen",
        "revendedor_cidade": "city",
        "revendedor_bairro": "neighborhood",
        "revendedor_uf": "state",
        "revendedor_idade": "age",
        "revendedor_dt_primeira_compra": "dt_first_purchase",
        "revendedor_segmentacao": "segment",
        "revendedor_dt_cadastro": "dt_signup",
        "revendedor_status_comercial": "status_com",
        "revendedor_origem_cadastro": "orig_signup",
        "revendedor_cod_supervisor": "supervisor_id",
        "revendedor_cod_coordenador": "coord_id",
        "revendedor_tipo_atendimento": "att_type",
        "revendedor_modelo_atendimento": "att_model",
        "revendedor_filial": "filial",
        "revendedor_praca": "praca",
        "revendedor_gr": "gr",
        "revendedor_supervisora": "supervisora",
        "revendedor_coordenadora": "coordenadora",
        "ciclos_inativos_atual_calculado": "inactive_cycles"
    }
    df = df.drop(columns=["revendedor_nome", "revendedor_email", "revendedor_telefone"])

    df = df.rename(columns=renames)

    df["rev_id"] = df["rev_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["rev_id"], keep="first")
    return df
