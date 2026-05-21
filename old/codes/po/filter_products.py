import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def load_active_products(path: str) -> set:
    """Load active product codes from a Parquet file."""
    df = pd.read_parquet(path)
    df["produto_cod"] = df["produto_cod"].astype(str)
    return set(df["produto_cod"].unique())

def load_and_filter_sales(base_dir: str, active_products: set) -> pd.DataFrame:
    """Load sales data from multiple partitions, filter by active products and remove gifts."""
    filtered_dfs = []
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
            df["produto_cod"] = df["produto_cod"].astype(str)

            # Filter by active products and exclude "Brinde" (gift)
            df_filtered = df[
                (df["produto_cod"].isin(active_products)) &
                (df["venda_tipo"] != "Brinde")
            ]

            if not df_filtered.empty:
                ciclo_value = partition_dir.split("=")[-1]
                df_filtered.loc[:, "venda_ciclo"] = ciclo_value
                filtered_dfs.append(df_filtered)

    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def save_partitioned_with_arrow(df: pd.DataFrame, output_path: str):
    """Save DataFrame as partitioned Parquet using PyArrow."""
    if df.empty:
        print("⚠️ No data to save.")
        return

    # Select only necessary columns
    selected_columns = [
        "venda_ciclo", "produto_cod", "revendedor_cod", "venda_data",
        "venda_qtd", "venda_vlr_receita_liquida", "venda_vlr_receita_bruta", "venda_tipo"
    ]
    df = df[selected_columns]

    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=output_path,
        partition_cols=["venda_ciclo"]
    )

    print(f"✅ Partitioned data saved at: {output_path}")

if __name__ == "__main__":
    sales_path = "data/raw/vendas.parquet"
    active_products_path = "data/clean/produtos_ativos.parquet"
    clean_output_path = "data/clean/vendas.parquet"

    active_products = load_active_products(active_products_path)
    filtered_df = load_and_filter_sales(sales_path, active_products)

    print(f"📦 Total rows with active products: {len(filtered_df)}")
    save_partitioned_with_arrow(filtered_df, clean_output_path)
