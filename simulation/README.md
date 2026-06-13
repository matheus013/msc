# simulation — Pipeline Kedro (AIPE)

Pipeline de simulação e seleção de políticas de inventário para a dissertação de mestrado.
Suporta qualquer estado brasileiro e qualquer produto com demanda intermitente,
com parâmetros totalmente configuráveis via YAML.

## Instalação

```bash
cd simulation
pip install -e . -r requirements.txt
```

## Execução

```bash
# Pipeline completo — configuração padrão (PB)
kedro run

# Fase 1: Paraíba
kedro run --params "data_ingestion.states=['PB']"

# Fase 2: Bahia
kedro run --params "data_ingestion.states=['BA']"

# Todos os estados com demanda intermitente
kedro run --params "data_ingestion.states=['all'],data_ingestion.products=null"

# Pipelines individuais
kedro run --pipeline demand_profiling
kedro run --pipeline statistical_validation

# DAG visual
kedro viz
```

## Pipelines (7)

| Pipeline | Descrição |
|---|---|
| `data_ingestion` | Carrega Parquet por UF, filtra, limpa, gera cenários por (estado, produto, loja) |
| `demand_profiling` | Classifica séries em Perfis Operacionais de Demanda (POD) via ADI × CV² |
| `demand_forecasting` | Treina modelos de previsão por loja (Croston, ARIMA, XGBoost, ANN) |
| `inventory_simulation` | Simula 12 políticas × todas as lojas × n réplicas |
| `policy_selection` | Gera rótulos e features para o Policy Selection Engine (PSE) |
| `statistical_validation` | Wilcoxon, Friedman-Nemenyi, Cohen's d com correção Bonferroni |
| `reporting` | Figuras PDF, tabelas LaTeX (booktabs) |

## Configuração

Edite os arquivos em `conf/base/parameters/`:

| Arquivo | Controla |
|---|---|
| `data_ingestion.yml` | Estado(s), produto(s), intervalo de datas |
| `data_cleaning.yml` | Regras de limpeza, outliers, missing |
| `demand_profiling.yml` | Limiares ADI, CV², burstiness, sazonalidade |
| `forecasting.yml` | Modelos, lookback, hiperparâmetros |
| `simulation.yml` | Políticas, custos, lead time, réplicas |
| `policy_selection.yml` | Critério de rótulo (NS mínimo), random seed |
| `reporting.yml` | Formato das figuras e tabelas |

## Políticas Avaliadas (12)

| Categoria | Políticas |
|---|---|
| Clássicas | EOQ, (s,S), Jornaleiro (Newsvendor) |
| Meta-heurísticas | GA, SA, PSO, DE |
| Aprendizado por Reforço | DQN, PPO, SARSA |
| Híbridas (AIPE) | GA-DQN, GA-PPO |

## KPIs (5)

| KPI | Descrição |
|---|---|
| **TIC** | Custo Total de Inventário (R$) |
| **NS** | Nível de Serviço (0–1) |
| **TR** | Taxa de Ruptura (0–1) |
| **BE** | Efeito Bullwhip (razão de variâncias) |
| **FP** | Frequência de Pedidos (0–1) |

## Estrutura do código

```
src/simulation/
├── core/
│   ├── inventory_env.py   ← ambiente de simulação (InventoryEnv)
│   ├── policies.py        ← 12 políticas de reposição
│   ├── forecasting.py     ← modelos de previsão
│   └── visualizations.py  ← helpers de visualização
└── pipelines/
    ├── data_ingestion/
    ├── demand_profiling/
    ├── demand_forecasting/
    ├── inventory_simulation/
    ├── policy_selection/
    ├── statistical_validation/
    └── reporting/
```
