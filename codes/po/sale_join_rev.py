import pandas as pd

sales = pd.read_parquet("data/clean/vendas.parquet")
rev = pd.read_parquet("data/clean/rev.parquet")

print("Sales shape:", sales.shape)
print("Rev shape:", rev.shape)

# Garantir que ambas as colunas estão como string
sales["revendedor_cod"] = sales["revendedor_cod"].astype(str)
rev["id"] = rev["id"].astype(str)

# Realiza o merge usando revendedor_cod (em sales) e id (em rev)
df_joined = sales.merge(rev, left_on='revendedor_cod', right_on='id', how='left')

print("Joined shape:", df_joined.shape)

print(df_joined.value_counts())
# df_joined.to_parquet(
#     "data/clean/vendas_uf",
#     index=False,
#     partition_cols=["uf"]
# )
