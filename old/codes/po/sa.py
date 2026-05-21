import json
import os
import time
import random
import math

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def parse_key(k):
    return tuple(k.split("|"))

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

def evaluate(solution, items, stores, transport_cost, prod_cost, demand, initial_stock):
    item_to_center = dict(zip(items, solution))
    total_cost = 0

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

    return total_cost

def run_sa_cycle(cycle_dir, runs=3, max_iter=500, initial_temp=1000, alpha=0.95):
    print(f"\n🔄 Starting SA for cycle: {os.path.basename(cycle_dir)}")
    transport_cost, prod_cost, demand, initial_stock, items, stores, centers = load_inputs(cycle_dir)

    for run in range(1, runs + 1):
        print(f"\n▶️ SA Run {run}/{runs}...")
        start_time = time.time()

        current = [random.choice(centers) for _ in items]
        current_cost = evaluate(current, items, stores, transport_cost, prod_cost, demand, initial_stock)
        best = current[:]
        best_cost = current_cost

        temp = initial_temp
        for i in range(max_iter):
            neighbor = current[:]
            idx = random.randint(0, len(items) - 1)
            neighbor[idx] = random.choice(centers)
            neighbor_cost = evaluate(neighbor, items, stores, transport_cost, prod_cost, demand, initial_stock)

            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            temp *= alpha
            if temp < 1e-3:
                break

            if i % 100 == 0:
                print(f"  Iter {i}: Best cost = {best_cost:.2f}")

        elapsed = round(time.time() - start_time, 4)
        solution = {
            "run": run,
            "cost": round(best_cost, 2),
            "elapsed_time_sec": elapsed,
            "assignment": {item: best[idx] for idx, item in enumerate(items)}
        }

        save_json(solution, os.path.join(cycle_dir, f"solution_sa_run_{run}.json"))
        print(f"✅ SA Run {run} saved | Cost: {solution['cost']} | Time: {elapsed}s")

def run_all_sa_cycles(base_dir="outputs", runs_per_cycle=3):
    cycle_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    for cycle_dir in cycle_dirs:
        run_sa_cycle(cycle_dir, runs=runs_per_cycle)

# Execute
run_all_sa_cycles(runs_per_cycle=5)
