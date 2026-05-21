import os
import json
import pandas as pd

# Lista para armazenar os dados
dados = []

# Itera sobre os anos e campanhas
for ano in range(2023, 2026):
    for campanha in range(1, 18):
        campanha_str = f"{campanha:02d}"
        pasta = f"outputs/{ano}{campanha_str}"
        caminho_arquivo = os.path.join(pasta, "demand.json")
        
        # Verifica se o arquivo demand.json existe
        if os.path.isfile(caminho_arquivo):
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                try:
                    conteudo = json.load(f)
                    # Itera sobre store_id e produto_id
                    for store_id, produtos in conteudo.items():
                        for produto_id, demanda in produtos.items():
                            dados.append({
                                'ano': ano,
                                'campanha': campanha,
                                'produto_id': produto_id,
                                'store_id': store_id,
                                'demanda': demanda
                            })
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON em: {caminho_arquivo}")
        else:
            pass

# Cria o DataFrame
df = pd.DataFrame(dados)

# Exibe as primeiras linhas do DataFrame
print(df.head())
print(df.shape)

df.to_parquet('data/raw/base.parquet')