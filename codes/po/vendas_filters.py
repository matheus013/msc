import os

import pandas as pd


def load_active_products() -> set:
    """Load active product codes from a Parquet file."""
    df = pd.read_parquet('data/raw/tb_revendedor.parquet')
    df = df[
        (df["revendedor_uf"] == "PB")
    ]
    df["revendedor_cod"] = df["revendedor_cod"].astype(str)
    return set(df["revendedor_cod"].unique())


def load_and_filter_sales(base_dir: str, active_products: set) -> pd.DataFrame:
    """Load sales data from multiple partitions, filter by active products and remove gifts."""
    total_rows = 0
    partitions = [d for d in os.listdir(base_dir) if d.startswith("venda_ciclo=")]

    for partition_dir in partitions:
        full_path = os.path.join(base_dir, partition_dir)
        parquet_files = [
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if f.endswith(".parquet")
        ]

        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            df["revendedor_cod"] = df["revendedor_cod"].astype(str)

            df_filtered = df[
                (df["revendedor_cod"].isin(active_products))
            ]
            total_rows += len(df_filtered)
            print(f"Total rows: {len(df_filtered)} in {file_path}")
    return total_rows

if __name__ == "__main__":
    sales_path = "data/raw/vendas.parquet"
    active_products_path = "data/clean/produtos_ativos.parquet"

    active_products = load_active_products()
    filtered_df = load_and_filter_sales(sales_path, active_products)

    print(f"📦 Total rows with active products: {filtered_df}")
