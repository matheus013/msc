import json
import os
import time
import psutil
from pulp import LpProblem, LpMinimize, LpVariable, lpSum


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_key(key):
    return tuple(key.split("|"))


def solve_cycle(cycle_dir, runs=3):
    print(f"\n🔄 Starting cycle: {os.path.basename(cycle_dir)}")

    # Load inputs
    transport_cost_raw = load_json(os.path.join(cycle_dir, "transport_cost.json"))
    production_cost = load_json(os.path.join(cycle_dir, "production_cost.json"))
    capacity_raw = load_json(os.path.join(cycle_dir, "capacity.json"))
    demand = load_json(os.path.join(cycle_dir, "demand.json"))
    initial_stock = load_json(os.path.join(cycle_dir, "initial_stock.json"))

    # Convert keys
    transport_cost = {parse_key(k): v for k, v in transport_cost_raw.items()}
    capacity = {parse_key(k): v for k, v in capacity_raw.items()}

    # Sets
    items = sorted({item for d in demand.values() for item in d})
    stores = sorted(demand.keys())
    storage_centers = sorted(initial_stock.keys())
    factories = list(production_cost.keys())

    for run in range(1, runs + 1):
        print(f"\n▶️ Run {run}/{runs} for cycle {os.path.basename(cycle_dir)}...")

        start_time = time.time()
        cpu_percent_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().used / (1024 ** 2)

        # Build model
        problem = LpProblem("Supply_Optimization", LpMinimize)

        x = {(a, b, i): LpVariable(f"x_{a}_{b}_{i}", lowBound=0) for (a, b) in transport_cost for i in items}
        y = {(f, i): LpVariable(f"y_{f}_{i}", lowBound=0) for f in factories for i in items}

        problem += (
            lpSum(transport_cost[a, b] * x[a, b, i] for (a, b) in transport_cost for i in items) +
            lpSum(production_cost[f][i] * y[f, i] for f in factories for i in items)
        )

        # Constraints
        for s in stores:
            for i in items:
                problem += lpSum(x[sc, s, i] for sc in storage_centers if (sc, s) in transport_cost) >= demand[s].get(i, 0)

        for sc in storage_centers:
            for i in items:
                entrada = lpSum(x[f, sc, i] for f in factories if (f, sc) in transport_cost)
                saida = lpSum(x[sc, s, i] for s in stores if (sc, s) in transport_cost)
                stock = initial_stock[sc].get(i, 0)
                problem += entrada + stock >= saida

        for f in factories:
            for i in items:
                problem += lpSum(x[f, sc, i] for sc in storage_centers if (f, sc) in transport_cost) == y[f, i]

        for (sc, s), cap in capacity.items():
            for i in items:
                problem += x[sc, s, i] <= cap

        # Solve
        problem.solve()

        # Monitor
        elapsed_time = round(time.time() - start_time, 4)
        cpu_percent_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().used / (1024 ** 2)

        print(f"⏱️  Time: {elapsed_time:.2f}s | 🧠 RAM: {mem_after - mem_before:.2f} MB | CPU: {(cpu_percent_before + cpu_percent_after) / 2:.2f}%")

        # Save results
        result = {
            "run": run,
            "objective_value": round(problem.objective.value(), 2),
            "elapsed_time_sec": elapsed_time,
            "cpu_percent": round((cpu_percent_before + cpu_percent_after) / 2, 2),
            "ram_used_mb": round(mem_after - mem_before, 2),
            "variables": {
                var.name: round(var.varValue, 2)
                for var in problem.variables()
                if var.varValue and var.varValue > 0
            }
        }

        out_path = os.path.join(cycle_dir, f"solution_run_{run}.json")
        save_json(result, out_path)
        print(f"💾 Solution saved to {out_path}")

    print(f"\n✅ Completed cycle: {os.path.basename(cycle_dir)}")


# ---------- MAIN ----------
base_dir = "outputs"
runs_per_cycle = 1  # change this as needed

cycle_dirs = sorted([
    os.path.join(base_dir, d)
    for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])
skip = list(range(202301, 202315))
for cycle_dir in cycle_dirs:
    if int(cycle_dir.split('\\')[1]) in skip:
        continue
    solve_cycle(cycle_dir, runs=runs_per_cycle)
