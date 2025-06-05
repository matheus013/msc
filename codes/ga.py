import json
import os
import time
import random
from deap import base, creator, tools, algorithms

# ---------- Utilitários ----------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def parse_key(k):
    return tuple(k.split("|"))

# ---------- Leitura dos dados ----------
def load_inputs(cycle_dir):
    transport_cost_raw = load_json(os.path.join(cycle_dir, "transport_cost.json"))
    production_cost = load_json(os.path.join(cycle_dir, "production_cost.json"))
    demand = load_json(os.path.join(cycle_dir, "demand.json"))
    initial_stock = load_json(os.path.join(cycle_dir, "initial_stock.json"))

    transport_cost = {parse_key(k): v for k, v in transport_cost_raw.items()}
    factories = list(production_cost.keys())
    items = sorted({item for d in demand.values() for item in d})
    stores = sorted(demand.keys())
    centers = sorted({k[0] for k in transport_cost.keys() if not k[0].startswith("factory")})

    return transport_cost, production_cost[factories[0]], demand, initial_stock, items, stores, centers

# ---------- Avaliação: 1 gene por item ----------
def evaluate(individual, items, stores, transport_cost, prod_cost, demand, initial_stock):
    item_to_center = dict(zip(items, individual))
    total_cost = 0

    # Estoque inicial por centro
    stock = {center: {item: initial_stock.get(center, {}).get(item, 0) for item in items} for center in set(item_to_center.values())}

    for store in stores:
        for item in items:
            center = item_to_center[item]
            qty = demand[store].get(item, 0)
            transport = transport_cost.get((center, store), 9999)
            total_cost += transport * qty

            available = stock[center].get(item, 0)
            to_produce = max(0, qty - available)
            stock[center][item] = max(0, available - qty)
            total_cost += prod_cost.get(item, 0) * to_produce

    return (total_cost,)

# ---------- Execução GA para um ciclo ----------
def run_ga_cycle(cycle_dir, runs=3, ngen=50, pop_size=80):
    print(f"\n🔄 Starting GA for cycle: {os.path.basename(cycle_dir)}")
    transport_cost, prod_cost, demand, initial_stock, items, stores, centers = load_inputs(cycle_dir)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: random.choice(centers), n=len(items))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, items=items, stores=stores, transport_cost=transport_cost,
                     prod_cost=prod_cost, demand=demand, initial_stock=initial_stock)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(centers)-1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    for run in range(1, runs + 1):
        print(f"\n▶️ Run {run}/{runs}...")

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        start_time = time.time()
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, halloffame=hof, verbose=True)
        elapsed = round(time.time() - start_time, 4)

        best = hof[0]
        total_cost = best.fitness.values[0]

        # Construir solução
        solution = {
            "run": run,
            "cost": round(total_cost, 2),
            "elapsed_time_sec": elapsed,
            "assignment": {item: best[idx] for idx, item in enumerate(items)}
        }

        save_json(solution, os.path.join(cycle_dir, f"solution_ga_run_{run}.json"))
        print(f"✅ Run {run} saved | Cost: {solution['cost']} | Time: {elapsed}s")

# ---------- Execução para todos os ciclos ----------
def run_all_ga_cycles(base_dir="outputs", runs_per_cycle=3):
    cycle_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    for cycle_dir in cycle_dirs:
        run_ga_cycle(cycle_dir, runs=runs_per_cycle)

# ---------- Iniciar ----------
run_all_ga_cycles(runs_per_cycle=5)
