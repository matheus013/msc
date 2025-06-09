import json
import os
import time
import random

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

def construct_greedy_randomized_solution(items, centers, alpha=0.3):
    solution = []
    for _ in items:
        costs = {c: random.random() for c in centers}
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        threshold = min_cost + alpha * (max_cost - min_cost)
        rcl = [c for c in centers if costs[c] <= threshold]
        solution.append(random.choice(rcl))
    return solution

def local_search(solution, items, stores, centers, transport_cost, prod_cost, demand, initial_stock):
    best = solution[:]
    best_cost = evaluate(best, items, stores, transport_cost, prod_cost, demand, initial_stock)
    improved = True

    while improved:
        improved = False
        for i in range(len(items)):
            for c in centers:
                if c == best[i]:
                    continue
                neighbor = best[:]
                neighbor[i] = c
                cost = evaluate(neighbor, items, stores, transport_cost, prod_cost, demand, initial_stock)
                if cost < best_cost:
                    best = neighbor
                    best_cost = cost
                    improved = True
    return best, best_cost

def run_grasp_cycle(cycle_dir, runs=3, iterations=50, alpha=0.3):
    print(f"\n🔄 Starting GRASP for cycle: {os.path.basename(cycle_dir)}")
    transport_cost, prod_cost, demand, initial_stock, items, stores, centers = load_inputs(cycle_dir)

    for run in range(1, runs + 1):
        print(f"\n▶️ GRASP Run {run}/{runs}...")
        start_time = time.time()

        best_solution = None
        best_cost = float("inf")

        for it in range(iterations):
            candidate = construct_greedy_randomized_solution(items, centers, alpha)
            improved, cost = local_search(candidate, items, stores, centers, transport_cost, prod_cost, demand, initial_stock)

            if cost < best_cost:
                best_solution = improved
                best_cost = cost

            if it % 10 == 0:
                print(f"  Iter {it}: Best cost so far = {best_cost:.2f}")

        elapsed = round(time.time() - start_time, 4)
        result = {
            "run": run,
            "cost": round(best_cost, 2),
            "elapsed_time_sec": elapsed,
            "assignment": {item: best_solution[idx] for idx, item in enumerate(items)}
        }

        save_json(result, os.path.join(cycle_dir, f"solution_grasp_run_{run}.json"))
        print(f"✅ GRASP Run {run} saved | Cost: {result['cost']} | Time: {elapsed}s")

def run_all_grasp_cycles(base_dir="outputs", runs_per_cycle=3):
    cycle_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    for cycle_dir in cycle_dirs:
        run_grasp_cycle(cycle_dir, runs=runs_per_cycle)

# Execute
run_all_grasp_cycles(runs_per_cycle=5)