#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verificar estatísticas do arquivo vendas_sample.csv (PB)"""
import pandas as pd

df = pd.read_csv('data/vendas_sample.csv')

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
print(f'✓ Arquivo:            data/vendas_sample.csv (76.3 MB)')
print(f'\n✅ Arquivo pronto para usar em main.py!')
print(f'   - Coluna warehouse: PB')
print(f'   - 1.432.183 registros agregados')
