# simulation — Pipeline Kedro para Dissertação de Mestrado

Pipeline de otimização de inventário multi-echelon usando Kedro.  
Suporta qualquer estado brasileiro, qualquer produto com demanda intermitente,  
com filtros de limpeza e parâmetros totalmente configuráveis via YAML.

## Instalação

```bash
cd simulation
pip install -r requirements.txt
pip install -e .
```

## Execução

```bash
# Pipeline completo com configuração padrão (PB, produto 48130)
kedro run

# Todos os estados, todos os produtos com CV >= 1.5
kedro run --params "data_ingestion.states=['all'],data_ingestion.products=null"

# Estado específico
kedro run --params "data_ingestion.states=['BA']"

# Só ingestão + simulação
kedro run --pipeline data_ingestion,inventory_simulation

# Só validação estatística (requer kpis.parquet já gerado)
kedro run --pipeline statistical_validation

# Visualizar DAG de dependências
kedro viz
```

## Estrutura dos Pipelines

| Pipeline | Descrição |
|---|---|
| `data_ingestion` | Carrega Parquet, filtra, limpa, gera cenários por (estado, produto, loja) |
| `demand_forecasting` | Treina LSTM / ANN / XGBoost por loja |
| `inventory_simulation` | 12 políticas × todas as lojas × n réplicas |
| `statistical_validation` | Wilcoxon, Friedman+Nemenyi, effect sizes |
| `reporting` | Figuras, tabelas LaTeX |

## Configuração

Edite os arquivos em `conf/base/parameters/`:

- `data_ingestion.yml` — estado(s), produto(s), intervalo de datas
- `data_cleaning.yml` — regras de limpeza (outliers, missing)
- `forecasting.yml` — arquiteturas ML, lookback
- `simulation.yml` — políticas, custos, réplicas, lead_time
- `reporting.yml` — formato de saída das figuras

## Políticas Avaliadas (12)

| Categoria | Políticas |
|---|---|
| Clássicas | EOQ, (s,S), Jornaleiro (Newsvendor) |
| Meta-heurísticas | GA, SA, PSO, DE |
| Aprendizado por Reforço | DQN, PPO, SARSA |
| Híbridas (arquitetura proposta) | GA-DQN, GA-PPO |

## KPIs (5)

- **TIC** — Custo Total de Inventário (R$)
- **NS** — Nível de Serviço (0–1)
- **TR** — Taxa de Ruptura (0–1)
- **BE** — Efeito Bullwhip (razão de variâncias)
- **FP** — Frequência de Pedidos (0–1)
