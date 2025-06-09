import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.read_parquet('data/raw/tb_revendedor.parquet')
df = df[['revendedor_cod',
         'revendedor_estrutura_comercial', 'revendedor_genero', 'revendedor_uf',
         'revendedor_idade', 'revendedor_segmentacao',
         'revendedor_status_comercial', 'revendedor_filial', 'revendedor_praca',
         'revendedor_gr', 'ciclos_inativos_atual_calculado']]

df = df.rename(columns={
    'revendedor_cod': 'id',
    'revendedor_estrutura_comercial': 'estrutura',
    'revendedor_genero': 'genero',
    'revendedor_uf': 'uf',
    'revendedor_idade': 'idade',
    'revendedor_segmentacao': 'segmento',
    'revendedor_status_comercial': 'status',
    'revendedor_filial': 'filial',
    'revendedor_praca': 'praca',
    'revendedor_gr': 'gerente_regional',
    'ciclos_inativos_atual_calculado': 'ciclos_inativos'
}).drop_duplicates()

# Filtro com todas as UFs válidas
df = df[df['uf'].isin([
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO',
    'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
    'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
])]

# Salvar particionado por UF
table = pa.Table.from_pandas(df)
pq.write_to_dataset(
    table,
    root_path='data/clean/rev.parquet',
    partition_cols=["uf"]
)
