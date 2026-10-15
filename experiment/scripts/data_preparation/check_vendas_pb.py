#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verificar estatísticas do arquivo vendas_sample.csv (PB)"""

# Mantem os caminhos locais ancorados no projeto ao executar este script.
import os as _os
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    _os.chdir(PROJECT_ROOT)

import pandas as pd

df = pd.read_csv('data/legacy/vendas_sample.csv')

print('\n╔════════════════════════════════════════════╗')
print('║  RESUMO DADOS DE PB (Paraíba)             ║')
print('╚════════════════════════════════════════════╝')
print(f'\n✓ Registros:          {len(df):,}')
print(f'✓ Warehouses:         {df["warehouse"].nunique()}')
print(f'✓ Produtos:           {df["item_id"].nunique():,}')
print(f'✓ PDVs:               {df["store_id"].nunique():,}')
print(f'✓ Períodos:           {df["venda_ciclo"].nunique()}')
print(f'\n✓ Demanda (un):       {df["sales"].sum():,.0f}')
print(f'✓ Receita:            R$ {df["revenue"].sum():,.2f}')
print(f'✓ Preço médio:        R$ {df["avg_price"].mean():.2f}')
print(f'\n✓ Data mínima:        {df["venda_ciclo"].min()}')
print(f'✓ Data máxima:        {df["venda_ciclo"].max()}')
print(f'✓ Arquivo:            data/legacy/vendas_sample.csv (76.3 MB)')
print(f'\n✅ Amostra pronta para preparar os cenarios de PB!')
print(f'   - Coluna warehouse: PB')
print(f'   - 1.432.183 registros agregados')
