import json
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# ---------- UTILITÁRIOS ----------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_key(key):
    return tuple(key.split("|"))

# ---------- LEITURA DE DADOS ----------

# Carregar parâmetros
transport_cost_raw = load_json("bases/202301/transport_cost.json")
production_cost = load_json("bases/202301/production_cost.json")
capacity_raw = load_json("bases/202301/capacity.json")
demand = load_json("bases/202301/demand.json")
initial_stock = load_json("bases/202301/initial_stock.json")

# Reconstruir dicionários com tupla como chave
transport_cost = {parse_key(k): v for k, v in transport_cost_raw.items()}
capacity = {parse_key(k): v for k, v in capacity_raw.items()}

# Derivar conjuntos
items = sorted({item for d in demand.values() for item in d})
stores = sorted(demand.keys())
storage_centers = sorted(initial_stock.keys())
factories = list(production_cost.keys())

# ---------- DEFINIÇÃO DO PROBLEMA ----------

problem = LpProblem("Supply_Optimization", LpMinimize)

# Variáveis de transporte e produção
x = {(a, b, i): LpVariable(f"x_{a}_{b}_{i}", lowBound=0) for (a, b) in transport_cost for i in items}
y = {(f, i): LpVariable(f"y_{f}_{i}", lowBound=0) for f in factories for i in items}

# ---------- FUNÇÃO OBJETIVO ----------

problem += (
    lpSum(transport_cost[a, b] * x[a, b, i] for (a, b) in transport_cost for i in items) +
    lpSum(production_cost[f][i] * y[f, i] for f in factories for i in items)
)

# ---------- RESTRIÇÕES ----------

# Atender a demanda das lojas
for s in stores:
    for i in items:
        problem += lpSum(x[sc, s, i] for sc in storage_centers if (sc, s) in transport_cost) >= demand[s].get(i, 0)

# Balanço nos centros de distribuição
for sc in storage_centers:
    for i in items:
        entrada = lpSum(x[f, sc, i] for f in factories if (f, sc) in transport_cost)
        saida = lpSum(x[sc, s, i] for s in stores if (sc, s) in transport_cost)
        stock = initial_stock[sc].get(i, 0)
        problem += entrada + stock >= saida

# Produção = envio da fábrica para os centros
for f in factories:
    for i in items:
        problem += lpSum(x[f, sc, i] for sc in storage_centers if (f, sc) in transport_cost) == y[f, i]

# (Opcional) Capacidade dos centros
for (sc, s), cap in capacity.items():
    for i in items:
        problem += x[sc, s, i] <= cap  # ou x[sc, s, i] <= cap[i] se for por item

# ---------- SOLUÇÃO ----------

problem.solve()

# ---------- RESULTADOS ----------

print("\n🔎 Resultados das variáveis:")
for var in problem.variables():
    if var.varValue and var.varValue > 0:
        print(f"{var.name} = {var.varValue}")

print(f"\n✅ Valor total da função objetivo: {problem.objective.value()}")
