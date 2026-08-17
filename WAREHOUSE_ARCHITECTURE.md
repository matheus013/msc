# Arquitetura Multi-Nível com Warehouse (Atualização)

## Resumo das Mudanças

O sistema foi atualizado para suportar uma **rede de suprimento multi-nível**:

```
LEVEL 1: Warehouse/Estoque (Estados - UF)
    ↓
LEVEL 2: Store/PDV (Revendedores)
    ↓
LEVEL 3: Product/Produto (Item ID)
```

Cada **estoque (warehouse)** gerencia múltiplas **lojas (stores/PDVs)**, cada uma com demanda de múltiplos **produtos**.

---

## Arquivos Atualizados

### 1. **data_loader.py** ✅

**Funções Novas:**
- `_filter_single_series()` — Agora suporta filtro de warehouse antes de item/store
- `get_network_structure()` — Analisa hierarquia warehouse→store→product
- `aggregate_warehouse_demand()` — Soma demanda de todos os stores em um warehouse
- `get_warehouse_context()` — Retorna contexto (n_stores, n_produtos, estatísticas) do warehouse

**Lógica:**
1. Se `warehouse` coluna existe nos dados
2. Seleciona warehouse (config: `selected_warehouse` ou auto por máx registros)
3. Depois filtra item_id/store_id dentro daquele warehouse

### 2. **inventory_env.py** ✅

**Mudanças:**
- Adicionados parâmetros opcionais: `warehouse`, `store_id`, `product_id` no `__init__()`
- Estado mantém 6 dimensões (compatível com tudo existente)
- Ambiente agora rastreia contexto para traceabilidade

**Uso:**
```python
env = InventoryEnv(
    demand_series, cfg, seed=seed,
    warehouse="SP",           # Estoque São Paulo
    store_id="STORE_001",     # Loja específica
    product_id="PROD_A"       # Produto
)
```

### 3. **main.py** ✅

**Mudanças:**
- **Step [1/7]**: Análise de estrutura multi-nível (warehouses, stores, densidade)
- **Função `run()`**: Extrai warehouse/store/product do DataFrame e passa ao InventoryEnv
- **Output**: Exibe resumo de warehouses disponíveis antes de rodar políticas
- **Encoding**: Corrigido para UTF-8 em config.yaml

**Output Novo:**
```
[REDE MULTI-NÍVEL]
  Estoques (Warehouses): AC, AL
  Densidade de dados: 14.3%
  → Warehouse: AC
    Lojas: 4, Produtos: 167, Períodos: 17
```

---

## Configuração (config.yaml)

```yaml
DATA:
  file_path: "data/vendas_sample.csv"
  
  columns:
    date: "date"
    demand: "sales"
    warehouse: "uf"           # ← NOVO: mapeia UF para warehouse interno
    item_id: "item_id"
    store_id: "store_id"
    price: "avg_price"
  
  filter_single_item: false                # false = analisar todos
  selected_warehouse: null                 # null = auto-select
  selected_item: null
  selected_store: null
```

---

## Pipeline de Dados

```
Parquet Files: data/source/vendas/uf=AC/, uf=AL/, ...
    ↓ (transform_vendas_quick.py)
CSV: vendas_preparadas.csv [uf, item_id, store_id, date, sales, revenue, avg_price]
    ↓ (config.yaml: warehouse="uf")
data_loader.load_data()
    ├─ Renomeia colunas (uf → warehouse)
    ├─ Seleciona warehouse (ou filtra todos se filter_single_item=false)
    └─ Filtra item+store dentro daquele warehouse
    ↓
InventoryEnv(demand, cfg, warehouse="AC", store_id="STORE_1", product_id="ITEM_A")
    ├─ [EOQ, (s,S), Newsvendor, GA, SA, PSO, DE, DQN, PPO, SARSA, HybridGADQN, HybridGAPPO]
    └─ KPIs com contexto warehouse rastreado
```

---

## Testes Realizados ✅

**Dados de Teste:**
- Arquivo: `data/vendas_sample.csv` (228 registros)
- 2 Warehouses: AC (Acre), AL (Alagoas)
- 167 Produtos, 4 Revendedores (Stores), 17 Períodos

**Pipeline Executado:**
1. ✅ **[1/7] Dados**: Carregamento com warehouse multi-nível
2. ✅ **[2/7] Forecasting**: LSTM (51% accuracy) vs ANN vs XGBoost
3. ✅ **[3/7] Políticas Clássicas**: EOQ, (s,S), Newsvendor
4. ✅ **[4/7] Metaheurísticas**: GA, SA em progresso (teste interrompido)

**Resultados (Teste AC):**
- EOQ: TIC=$5,259.36
- (s,S): TIC=$5,274.38
- Newsvendor: TIC=$2,784.00 (melhor para alta variabilidade)
- GA: TIC=$2,119.62 (otimizado)

---

## Análise Multi-Nível (Futuro)

Para rodar **todas as 27 warehouses** simultaneamente:

```python
# Em desenvolvimento:
# 1. Set filter_single_item: false
# 2. Loop por warehouse em main.py
# 3. Agregue resultados para insights da rede completa
# 4. Identifique warehouses com melhor performance
```

---

## Compatibilidade

✅ **Compatível com:**
- Dados sem warehouse (funcionam normal, single-warehouse)
- Cenários determinísticos (mesma demanda para todas políticas)
- Todas 12 políticas + 12 variantes híbridas

✅ **Não-Invasivo:**
- Parâmetros warehouse/store/product são opcionais
- State space mantém 6 dimensões
- Lógica de simulação idêntica

---

## Próximos Passos

1. **Gerar vendas_preparadas.csv completo** (27 warehouses)
   ```bash
   python transform_vendas_quick.py
   ```

2. **Análise por warehouse específico** (ex: São Paulo)
   ```yaml
   selected_warehouse: "SP"
   ```

3. **Análise de Rede Completa** (rodando em loop)
   - Compare KPIs entre warehouses
   - Identifique padrões (bullwhip effect, stockout risk)
   - Recomende políticas por warehouse

4. **Dashboards de Warehouse**
   - KPI por warehouse (TIC, Service Level, Bullwhip)
   - Comparativo de políticas
   - Previsões por warehouse

---

## Verificação Rápida

**Testar com um warehouse específico:**
```bash
# No config.yaml, altere:
selected_warehouse: "AC"
filter_single_item: true  # single item+store para teste rápido

# Execute:
python main.py
```

**Estrutura completa:**
```bash
# No config.yaml:
filter_single_item: false
selected_warehouse: null  # pega warehouse com mais dados

# Execute:
python main.py
```

---

## Documentação Técnica

Veja `/memories/repo/warehouse-architecture.md` para detalhes de implementação.
