import pandas as pd

for uf in [
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO',
    'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
    'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
]:
    try:
        df = pd.read_parquet(f"data/clean/vendas_uf/uf={uf}")

        print(uf, df.shape)
    except FileNotFoundError:
        pass
