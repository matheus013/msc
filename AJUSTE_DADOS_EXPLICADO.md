# Ajuste de Dados para Pipeline Zabraoui et al. (2025)

## Resumo do que foi feito

### 1. Transformação de Dados (JSON → CSV)
**Arquivo criado:** `transform_data_simple.py`

Converteu seus dados de:
- **Origem:** 38 arquivos JSON estruturados hierarquicamente
  - `demand.json`: {produto → loja → quantidade}
  - `capacity.json`: {loja|produto → capacidade}
  - `initial_stock.json`: {loja → {produto → estoque}}
  - `production_cost.json`: {fábrica → {produto → custo}}

- **Destino:** `data/dados_preparados.csv` (formato tabular)
  - **38 períodos** de jan/2023 a abr/2025
  - **Agregação:** Soma de demanda de todas as lojas/produtos
  - **Estatísticas:**
    - Demanda total: 763.460 unidades
    - Demanda média: 20.091 unidades/período
    - Variação: 7.435 a 46.455 unidades
    - Custo médio de produção: $20.06

### 2. Configuração Atualizada

**Arquivo:** `config.yaml` (modificado)

```yaml
DATA:
  file_path: "data/dados_preparados.csv"  # ← Novo arquivo
  columns:
    demand: "sales"
    price: "production_cost"              # ← Mapeado
  filter_single_item: false               # ← Desativado (dados já agregados)
  train_ratio: 0.7                        # ← Ajustado para 70% treino

COST:
  holding_cost_per_unit: 1.0
  stockout_cost_per_unit: 5.0
  ordering_cost_per_order: 50.0
  ordering_cost_per_unit: 0.5

SIMULATION:
  target_service_level: 0.95
  lead_time: 2
  initial_inventory: 100
  n_replications: 5
  random_seed: 42
```

---

## Como Usar

### Teste rápido (já pronto):
```bash
python main.py
```

Isso vai:
1. ✓ Carregar `dados_preparados.csv`
2. ✓ Treinar previsores (LSTM, XGBoost, ANN)
3. ✓ Executar 12 políticas (EOQ, (s,S), GA, SA, PSO, DE, DQN, PPO, Híbridas)
4. ✓ Gerar gráficos comparativos em `outputs/`

### Se precisar regenerar o CSV:
```bash
python transform_data_simple.py
# Escolha uma opção:
# [1] Soma — demanda agregada (RECOMENDADO - já usa)
# [2] Desagregado — item+loja completo (372.117 registros)
# [3] Primeira loja — exemplo para teste rápido
```

---

## Próximos Passos Recomendados

### Opção A: Executar com dados agregados (AGORA)
```bash
python main.py
```
✓ **Vantagem:** Rápido, analisa tendência geral
✓ **Desvantagem:** Perde nuances por item/loja

### Opção B: Trabalhar com dados desagregados (DETALHADO)

1. Regenere o CSV com opção [2]:
   ```bash
   python transform_data_simple.py
   # Escolha: 2
   ```

2. Modifique `config.yaml`:
   ```yaml
   filter_single_item: true
   selected_item: null      # Pega o primeiro
   selected_store: null
   train_ratio: 0.8
   ```

3. Execute:
   ```bash
   python main.py
   ```

---

## Estrutura de Arquivos Agora

```
sbpo/
├── main.py                          ← Script principal
├── config.yaml                      ← MODIFICADO
├── transform_data_simple.py         ← NOVO (transformador)
├── data/
│   ├── dados_preparados.csv         ← NOVO (38 períodos agregados)
│   └── scenario/                    ← Dados originais (JSON)
├── outputs/                         ← Gerado automaticamente
│   ├── comparison_report.csv        ← Resultados das 12 políticas
│   ├── fig_*.png                    ← Gráficos (Figs. 3-10)
│   └── ...
└── modules/
    ├── data_loader.py
    ├── forecasting.py
    ├── inventory_env.py
    ├── policies_extended.py
    ├── rl_agents.py
    └── visualizations.py
```

---

## Colunas do CSV Preparado

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `date` | data | Data do período (primeiro dia do mês) |
| `item_id` | str | "TOTAL" (agregado) |
| `store_id` | str | "ALL" (agregado) |
| `sales` | int | Demanda total do período |
| `production_cost` | float | Custo médio de produção |
| `initial_stock` | int | Estoque inicial (0 para agregado) |
| `capacity` | int | Capacidade total disponível |

---

## Opções Avançadas

### Se quiser usar dados por item específico:
```bash
# Regenere com opção [2] (desagregado)
python transform_data_simple.py 2

# Edite config.yaml para filtrar um item:
filter_single_item: true
selected_item: "10017690"    # Adicione o ID que quer
selected_store: null
```

### Se quiser adicionar mais dados:
1. Adicione novas pastas em `data/scenario/YYYYMM/`
2. Execute `transform_data_simple.py` novamente
3. O script detecta automaticamente novos períodos

---

## Troubleshooting

### "ModuleNotFoundError: pandas"
```bash
python -m pip install pandas numpy pyyaml scikit-learn xgboost
```

### CSV vazio
- Verifique se `data/scenario/` tem arquivos JSON
- Confirme que JSONs têm chaves `demand.json`, `capacity.json`, etc.

### Demanda com valores muito altos
- É esperado (soma de múltiplas lojas)
- Para desagregar: rode `transform_data_simple.py` com opção [2] ou [3]

---

**Criado:** Abril 27, 2026  
**Versão:** 1.0 - Transform Data  
**Status:** ✅ Pronto para executar `python main.py`
