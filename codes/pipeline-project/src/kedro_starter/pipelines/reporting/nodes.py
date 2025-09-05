from __future__ import annotations
import pandas as pd
from typing import List

def enrich_venda(vendas: pd.DataFrame, revendedor: pd.DataFrame, produto: pd.DataFrame) -> pd.DataFrame:
    """
    Build an enriched sales fact table:
    - Select only the requested columns from each source.
    - Normalize key types.
    - Join sales with resellers (revendedor) and products (produto).
    - Return with simplified/shorter column names.
    """

    # --- Columns requested by you ---
    r_columns: List[str] = [
        "rev_id", "city", "state", "segment", "att_type", "filial", "praca", "gr"
    ]
    p_columns: List[str] = [
        "produto_cod", "produto_grupo", "produto_categoria",
        "produto_subcategoria", "produto_condicao", "produto_marca"
    ]
    v_columns: List[str] = [
        "venda_data", "venda_qtd", "venda_vlr_receita_bruta",
        "venda_vlr_receita_liquida", "venda_meio_entrega",
        "venda_meio_captacao", "venda_ciclo", "produto_cod", "revendedor_cod"
    ]

    # --- Keep only requested columns that actually exist ---
    vendas = vendas[[c for c in v_columns if c in vendas.columns]].copy()
    revendedor = revendedor[[c for c in r_columns if c in revendedor.columns]].copy()
    produto = produto[[c for c in p_columns if c in produto.columns]].copy()

    # --- Normalize join keys as strings (robust merges) ---
    if "produto_cod" in vendas.columns:
        vendas["produto_cod"] = vendas["produto_cod"].astype("string").str.strip()
    if "revendedor_cod" in vendas.columns:
        vendas["revendedor_cod"] = vendas["revendedor_cod"].astype("string").str.strip()

    if "produto_cod" in produto.columns:
        produto["produto_cod"] = produto["produto_cod"].astype("string").str.strip()
    if "rev_id" in revendedor.columns:
        revendedor["rev_id"] = revendedor["rev_id"].astype("string").str.strip()

    # --- Merge sales + reseller (revendedor_cod -> rev_id) ---
    # We keep the reseller's rev_id as the canonical id and drop sales.revendedor_cod afterwards.
    if {"revendedor_cod"}.issubset(vendas.columns) and {"rev_id"}.issubset(revendedor.columns):
        df = vendas.merge(
            revendedor,
            left_on="revendedor_cod",
            right_on="rev_id",
            how="left",
            suffixes=("", "_rev")  # avoid collisions
        )
    else:
        # If the key is missing, just carry vendas forward.
        df = vendas.copy()

    # --- Merge with product on produto_cod ---
    if "produto_cod" in df.columns and "produto_cod" in produto.columns:
        df = df.merge(
            produto,
            on="produto_cod",
            how="left",
            suffixes=("", "_prod")
        )

    # --- Drop redundant key (we keep rev_id) ---
    if "revendedor_cod" in df.columns and "rev_id" in df.columns:
        df = df.drop(columns=["revendedor_cod"])

    # --- Final column simplification (snake_case, short names) ---
    # Keep consistent snake_case per PEP 8 for readability. :contentReference[oaicite:1]{index=1}
    rename_map = {
        # sales
        "venda_data": "date",
        "venda_qtd": "qty",
        "venda_vlr_receita_bruta": "rev_gross",
        "venda_vlr_receita_liquida": "rev_net",
        "venda_meio_entrega": "delivery_channel",
        "venda_meio_captacao": "capture_channel",
        "venda_ciclo": "cycle",
        "produto_cod": "prod_id",
        "rev_id": "rev_id",

        # reseller
        "city": "city",
        "state": "state",
        "segment": "segment",
        "att_type": "att_type",
        "filial": "branch",
        "praca": "plaza",
        "gr": "gr",

        # product
        "produto_grupo": "prod_group",
        "produto_categoria": "prod_category",
        "produto_subcategoria": "prod_subcategory",
        "produto_condicao": "prod_condition",
        "produto_marca": "prod_brand",
    }

    # Only rename columns that exist in df
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Optional: column order (sales core -> reseller -> product)
    ordered_cols = [
        # sales core
        "date", "cycle", "qty", "rev_gross", "rev_net", "delivery_channel", "capture_channel",
        # keys
        "rev_id", "prod_id",
        # reseller
        "city", "state", "segment", "att_type", "branch", "plaza", "gr",
        # product
        "prod_group", "prod_category", "prod_subcategory", "prod_condition", "prod_brand",
    ]
    # Keep only those present, plus any extras at the end
    final_cols = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]
    df = df[final_cols]

    return df
