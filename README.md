# Inventory Analytics — Reprodução de Zabraoui et al. (2025)

> **"A comparative study of multi-algorithm optimization for inventory analytics in supply chains"**
> Supply Chain Analytics, 12 (2025) 100154

Este projeto reimplementa o pipeline completo do artigo usando **sua própria base de dados**.

---

## Estrutura do Projeto

```
inventory_project/
├── config.yaml                  ← ✅ EDITE AQUI (seus dados e parâmetros)
├── main.py                      ← Script principal
├── generate_sample_data.py      ← Gera dados sintéticos para teste
├── data/
│   └── seus_dados.csv           ← Coloque sua base aqui
├── outputs/                     ← Gráficos e CSVs gerados automaticamente
└── modules/
    ├── data_loader.py           ← Carregamento e pré-processamento
    ├── forecasting.py           ← LSTM (numpy), ANN, XGBoost
    ├── inventory_env.py         ← Ambiente de simulação de estoque
    ├── policies.py              ← Heurística + Algoritmo Genético
    ├── rl_agents.py             ← DQN + PPO + Híbridos GA-DQN / GA-PPO
    └── visualizations.py        ← Todos os gráficos (Figs. 3–10 do artigo)
```

---

## Instalação

```bash
# Instalar dependências
pip install pandas numpy scikit-learn xgboost deap matplotlib seaborn pyyaml
```

---

## Como Usar

### 1. Teste rápido com dados sintéticos

```bash
cd inventory_project
python generate_sample_data.py    # Gera data/sample_data.csv
python main.py                    # Executa o pipeline completo
```

### 2. Com sua própria base de dados

**Passo 1** — Coloque seu arquivo em `data/` (CSV, Excel ou Parquet).

**Passo 2** — Edite `config.yaml`:

```yaml
DATA:
  file_path: "data/NOME_DO_SEU_ARQUIVO.csv"
  file_format: "csv"       # "csv", "excel" ou "parquet"
  columns:
    date: "sua_coluna_data"       # nome da coluna de data
    demand: "sua_coluna_vendas"   # nome da coluna de demanda ← OBRIGATÓRIO
    item_id: "sua_coluna_item"    # null se não houver
    store_id: "sua_coluna_loja"   # null se não houver
    price: "sua_coluna_preco"     # null se não houver
```

**Passo 3** — Execute:

```bash
python main.py
# ou com config alternativa:
python main.py --config config_meu_produto.yaml
```

---

## O que o Pipeline Faz

| Etapa | Módulo | O que faz |
|-------|--------|-----------|
| 1 | `data_loader.py` | Carrega dados, filtra item/loja, divide treino/teste |
| 2 | `forecasting.py` | Treina LSTM, ANN e XGBoost; calcula MAE, RMSE, MAPE, Accuracy |
| 3 | `policies.py` | Política heurística (ROP fixo) como baseline |
| 4 | `policies.py` | GA otimiza (ROP, Q, SS) via DEAP |
| 5 | `rl_agents.py` | DQN e PPO aprendem políticas dinâmicas |
| 5b | `rl_agents.py` | Híbridos GA-DQN e GA-PPO |
| 6 | `visualizations.py` | Gera Figs. 3–10 do artigo + tabela CSV |

---

## Saídas Geradas

Em `outputs/`:

| Arquivo | Descrição |
|---------|-----------|
| `fig3_forecast_comparison.png` | LSTM vs ANN vs XGBoost (Fig. 3) |
| `fig4_ppo_policy.png` | Execução da política PPO (Fig. 4) |
| `fig5_dqn_policy.png` | Simulação da política DQN (Fig. 5) |
| `fig6_heuristic_policy.png` | Gestão heurística (Fig. 6) |
| `fig7_ga_policy.png` | Convergência GA + política (Fig. 7) |
| `fig9_hybrid_overview.png` | GA-PPO e GA-DQN (Fig. 9) |
| `fig10_comparison_boxplots.png` | Boxplots comparativos (Fig. 10) |
| `table_kpi_summary.png` | Tabela resumo de KPIs |
| `kpi_inventory.csv` | KPIs de todas as políticas (CSV) |
| `kpi_forecasting.csv` | Métricas de previsão (CSV) |
| `ga_parameters.csv` | Parâmetros otimizados pelo GA |

---

## KPIs Calculados (igual ao artigo)

| KPI | Descrição |
|-----|-----------|
| **TIC** | Total Inventory Cost = holding + stockout + ordering |
| **Service Level** | % da demanda atendida (meta: 0.95) |
| **Stockout Rate** | % de períodos com falta de estoque |
| **Bullwhip Effect** | Var(pedidos) / Var(demanda) — volatilidade na cadeia |
| **Order Frequency** | Pedidos / Períodos |
| **MAE / RMSE / MAPE** | Acurácia de previsão |

---

## Ajustes Finos

### Melhorar acurácia do LSTM

```yaml
FORECASTING:
  lookback: 28        # mais janela histórica
  LSTM:
    hidden_size: 128
    epochs: 100
    learning_rate: 0.005
```

### GA mais rigoroso

```yaml
GENETIC_ALGORITHM:
  population_size: 200
  n_generations: 100
```

### DQN mais treinado

```yaml
DQN:
  episodes: 1000
  hidden_layers: [256, 128, 64]
```

### Múltiplos produtos

Defina `filter_single_item: false` no config e adapte `data_loader.py`
para agregar demanda por data antes de passá-la ao pipeline.

---

## Dependências

```
pandas >= 1.5
numpy >= 1.24
scikit-learn >= 1.2
xgboost >= 1.7
deap >= 1.3
matplotlib >= 3.6
seaborn >= 0.12
pyyaml >= 6.0
```
