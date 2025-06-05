import json
import pandas as pd
import numpy as np
import os

def to_json_compatible(obj):
    if isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj

def save_json(data, name, cycle, output_dir="bases"):
    path = os.path.join(output_dir, str(cycle))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{name}.json"), "w") as f:
        json.dump(to_json_compatible(data), f, indent=2)

def process_cycle(df_cycle, cycle):
    df_filtered = df_cycle[df_cycle["quantity"] > 0]
    df_best = (
        df_filtered.sort_values(["product_id", "quantity"])
        .drop_duplicates("product_id")
        .copy()
    )
    df_best["unit_net"] = df_best["net_revenue"] / df_best["quantity"]
    df_best["unit_gross"] = df_best["gross_revenue"] / df_best["quantity"]
    df_best["unit_base"] = df_best["unit_net"] / 1.6

    pricing = df_best.set_index("product_id")["unit_base"].round(2).to_dict()
    stores = df_cycle["store_id"].unique().tolist()
    products = df_cycle["product_id"].unique().tolist()
    centers = df_cycle["warehouse_id"].unique().tolist()

    demand = {}

    for (store, item), group in df_cycle.groupby(["store_id", "product_id"]):
        demand.setdefault(store, {})[item] = int(group['quantity'].sum())

    stock = {c: {p: 100 for p in products} for c in centers}
    transport = {f"{c}|{s}": 10 for c in centers for s in stores}
    production = {"factory1": pricing}
    capacity = {f"{c}|{s}": 1000 for c in centers for s in stores}

    save_json(demand, "demand", cycle)
    save_json(stock, "initial_stock", cycle)
    save_json(transport, "transport_cost", cycle)
    save_json(production, "production_cost", cycle)
    save_json(capacity, "capacity", cycle)

    return {
        "cycle": cycle,
        "products": len(products),
        "stores": len(stores),
        "warehouses": len(centers),
        "total_net": round(df_cycle["net_revenue"].sum(), 2),
        "total_gross": round(df_cycle["gross_revenue"].sum(), 2),
    }

# --------- MAIN ---------
uf = "MA"
df = pd.read_parquet(f"data/clean/vendas_uf/uf={uf}")
df = df.rename(columns={
    "produto_cod": "product_id",
    "revendedor_cod": "store_id",
    "venda_data": "date",
    "venda_qtd": "quantity",
    "venda_vlr_receita_liquida": "net_revenue",
    "venda_vlr_receita_bruta": "gross_revenue",
    "venda_tipo": "sale_type",
    "venda_ciclo": "cycle",
    "filial": "warehouse_id",
    "praca": "region"
})

print(df['region'].value_counts())

# df = df[df["region"] == "TER"]
df[["quantity"]] = df[["quantity"]].astype(int)
df[["net_revenue", "gross_revenue"]] = df[["net_revenue", "gross_revenue"]].astype(float)

summary = []
for cycle in df["cycle"].unique():
    result = process_cycle(df[df["cycle"] == cycle], cycle)
    summary.append(result)
    print(f"✅ Files saved for cycle {cycle}")

# Final report
report = pd.DataFrame(summary)
print("\n📊 Summary by cycle:")
print(report.to_string(index=False))
