from pulp import LpMinimize, LpProblem, LpVariable, lpSum

# Criar o modelo
model = LpProblem("Supply_Chain_Optimization", LpMinimize)

# Parâmetros de exemplo
S = {"S1", "S2"}  # Centros de estoque
M = {"M1", "M2"}  # Lojas
I = {"P1", "P2"}  # Itens
R = list(range(1, 18))  # Rounds (1 a 17)
d = {("S1", "M1"): 10, ("S1", "M2"): 15, ("S2", "M1"): 20, ("S2", "M2"): 5}  # Distâncias
c = 1  # Custo por unidade de distância
C = {("S1", "M1"): 100, ("S1", "M2"): 100, ("S2", "M1"): 100, ("S2", "M2"): 100}  # Capacidade máxima de transporte
O = {("M1", "P1"): {r: 10 for r in R}, ("M2", "P2"): {r: 150 for r in R}}  # Demanda esperada
E = {("S1", "P1"): 100, ("S2", "P2"): 100}  # Estoque inicial

# Variáveis de decisão
x = {
    (a, b, i, r): LpVariable(f"x_{a}_{b}_{i}_{r}", lowBound=0)
    for a in S | M for b in S | M for i in I for r in R
}

# Variáveis para o estoque restante
stock = {
    (s, i, r): LpVariable(f"stock_{s}_{i}_{r}", lowBound=0)
    for s in S for i in I for r in R
}

# Função Objetivo: minimizar o custo total do transporte
model += lpSum(c * d.get((a, b), 0) * x[a, b, i, r] for a in S | M for b in S | M for i in I for r in R)

# Restrição: Disponibilidade no centro de estoque
for s in S:
    for i in I:
        model += stock[s, i, 1] == E.get((s, i), 0)  # Estoque inicial no round 1
        for r in range(1, 17):
            model += stock[s, i, r+1] == stock[s, i, r] - lpSum(x[s, b, i, r] for b in S | M)  # Estoque reduzido após cada round

# Restrição: Atendimento da demanda da loja
for m in M:
    for i in I:
        for r in R:
            next_r = 1 if r == 17 else r + 1  # Ciclo dos rounds
            model += lpSum(x[a, m, i, r] for a in S | M) >= O.get((m, i), {}).get(next_r, 0)

# Restrição: Capacidade de transporte
for (a, b) in C.keys():
    for r in R:
        model += lpSum(x[a, b, i, r] for i in I) <= C[a, b]

# Resolver o modelo
model.solve()

# Verificar se o modelo foi resolvido com sucesso
if model.status == 1:  # 1 significa que foi solucionado com sucesso
    # Exibir resultados round a round
    for r in R:
        print(f"Round {r}:")
        
        # Exibir transporte de itens
        for (a, b, i, round_num) in x:
            if round_num == r:  # Garantir que estamos no round correto
                if x[a, b, i, r].varValue is not None and x[a, b, i, r].varValue > 0:
                    print(f"\tTransportar {x[a, b, i, r].varValue} unidades de {i} de {a} para {b}")

        # Exibir estoque restante
        for (s, i, round_num) in stock:
            if round_num == r:  # Garantir que estamos no round correto
                if stock[s, i, r].varValue is not None:
                    print(f"\tEstoque restante de {i} no {s}: {stock[s, i, r].varValue}")

        print("-" * 40)
else:
    print("O modelo não foi resolvido com sucesso.")
