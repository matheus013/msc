"""
generate_sample_data.py
Gera uma base de dados sintética similar ao Walmart M5 para testar o pipeline.
Execute: python generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Parâmetros
N_STORES = 3
N_ITEMS  = 5
N_DAYS   = 500

dates  = pd.date_range("2021-01-01", periods=N_DAYS, freq="D")
stores = [f"STORE_{i+1}" for i in range(N_STORES)]
items  = [f"ITEM_{i+1}" for i in range(N_ITEMS)]

rows = []
for store in stores:
    for item in items:
        # Nível base de demanda por item
        base_demand = np.random.uniform(20, 80)
        # Tendência suave
        trend = np.linspace(0, np.random.uniform(-10, 10), N_DAYS)
        # Sazonalidade semanal
        weekly = 10 * np.sin(np.arange(N_DAYS) * 2 * np.pi / 7)
        # Sazonalidade anual
        annual = 15 * np.sin(np.arange(N_DAYS) * 2 * np.pi / 365)
        # Ruído
        noise = np.random.normal(0, 5, N_DAYS)
        # Eventos aleatórios (picos de demanda)
        events = np.zeros(N_DAYS)
        event_days = np.random.choice(N_DAYS, size=10, replace=False)
        events[event_days] = np.random.uniform(20, 60, 10)
        # Preço variável
        price = np.random.uniform(5, 20) + np.random.normal(0, 0.5, N_DAYS)

        demand = base_demand + trend + weekly + annual + noise + events
        demand = np.clip(demand, 0, None).round().astype(int)

        for t, d in enumerate(dates):
            rows.append({
                "date":       d.strftime("%Y-%m-%d"),
                "store_id":   store,
                "item_id":    item,
                "sales":      demand[t],
                "sell_price": round(price[t], 2),
            })

df = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
path = "data/sample_data.csv"
df.to_csv(path, index=False)

print(f"Dados gerados: {len(df):,} linhas → {path}")
print(f"Período: {df['date'].min()} a {df['date'].max()}")
print(f"Itens: {df['item_id'].nunique()} | Lojas: {df['store_id'].nunique()}")
print(f"\nPrimeiras linhas:")
print(df.head(10).to_string(index=False))
print(f"\nEstatísticas de demanda:")
print(df["sales"].describe().round(2))
