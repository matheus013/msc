from pulp import LpProblem, LpMinimize, LpVariable, lpSum

# Define sets
items = ['item1', 'item2', 'item3', 'item4', 'item5']
storage_centers = ['center1', 'center2', 'center3']
factories = ['factory1']
stores = ['store1', 'store2', 'store3', 'store4']

# Parameters
transport_cost = {('factory1', 'center1'): 10, ('factory1', 'center2'): 15, ('factory1', 'center3'): 20,
                  ('center1', 'store1'): 5, ('center1', 'store2'): 7, ('center2', 'store3'): 6, ('center3', 'store4'): 8}
production_cost = {'factory1': {'item1': 3, 'item2': 4, 'item3': 5, 'item4': 6, 'item5': 7}}
capacity = {('center1', 'store1'): 50, ('center1', 'store2'): 60, ('center2', 'store3'): 55, ('center3', 'store4'): 65}
demand = {'store1': {'item1': 20, 'item2': 15, 'item3': 10, 'item4': 12, 'item5': 18},
          'store2': {'item1': 18, 'item2': 20, 'item3': 12, 'item4': 14, 'item5': 16},
          'store3': {'item1': 22, 'item2': 14, 'item3': 16, 'item4': 18, 'item5': 20},
          'store4': {'item1': 25, 'item2': 17, 'item3': 14, 'item4': 20, 'item5': 22}}
initial_stock = {'center1': {'item1': 50, 'item2': 40, 'item3': 30, 'item4': 20, 'item5': 25},
                 'center2': {'item1': 40, 'item2': 30, 'item3': 20, 'item4': 10, 'item5': 15},
                 'center3': {'item1': 30, 'item2': 25, 'item3': 15, 'item4': 10, 'item5': 20}}

# Define problem
problem = LpProblem("Supply_Optimization", LpMinimize)

# Decision variables
x = {(a, b, i): LpVariable(f"x_{a}_{b}_{i}", lowBound=0) for a, b in transport_cost.keys() for i in items}
y = {(f, i): LpVariable(f"y_{f}_{i}", lowBound=0) for f in factories for i in items}

# Objective function
problem += lpSum(transport_cost[a, b] * x[a, b, i] for a, b in transport_cost.keys() for i in items) + \
           lpSum(production_cost[f][i] * y[f, i] for f in factories for i in items)

# Constraints
for s in stores:
    for i in items:
        problem += lpSum(x[sc, s, i] for sc in storage_centers if (sc, s) in transport_cost) >= demand[s][i]

for sc in storage_centers:
    for i in items:
        problem += lpSum(x[f, sc, i] for f in factories if (f, sc) in transport_cost) + \
                   initial_stock[sc][i] >= lpSum(x[sc, s, i] for s in stores if (sc, s) in transport_cost)

for f in factories:
    for i in items:
        problem += lpSum(x[f, sc, i] for sc in storage_centers if (f, sc) in transport_cost) == y[f, i]

# Solve problem
problem.solve()

# Print results
for var in problem.variables():
    print(f"{var.name} = {var.varValue}")
