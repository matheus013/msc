# Otimização de Inventário Multi-Echelon com Aprendizado de Máquina

Repositório da pesquisa de mestrado e artigo SBPO 2026 sobre otimização de políticas de gestão de inventário em redes warehouse→loja, com demanda intermitente e dados reais de varejo regional brasileiro.

---

## Estrutura

```
sbpo/
├── docs/
│   ├── sbpo/                  ← Artigo SBPO 2026 (LaTeX)
│   │   ├── main.tex
│   │   ├── exemplo-latex.bib
│   │   ├── sbpo-template.sty
│   │   └── figuras/           ← Figuras geradas pelo experimento PB
│   ├── master_proposal/       ← Proposta de dissertação de mestrado (LaTeX)
│   └── references/            ← PDFs de referência aprovados
│
├── data/
│   ├── source/                ← Dados brutos proprietários (gitignored)
│   │   └── vendas/uf=*/       ← Parquet particionado por estado (27 UFs)
│   └── scenario/              ← Cenários por ciclo bimestral (gitignored)
│
└── simulation/                ← Pipeline Kedro — dissertação de mestrado
    ├── conf/base/             ← Parâmetros configuráveis por YAML
    ├── src/simulation/
    │   ├── core/              ← InventoryEnv, 12 políticas, forecasters
    │   └── pipelines/         ← 5 pipelines Kedro
    └── requirements.txt
```

---

## Artigo SBPO 2026

**"Otimização Multi-Echelon de Inventário com Demanda Intermitente: Uma Arquitetura Híbrida GA-RL"**

Avalia 12 políticas de inventário — clássicas (EOQ, (s,S), Newsvendor), metaheurísticas (GA, SA, PSO, DE) e aprendizado por reforço (DQN, PPO, SARSA, GA-DQN, GA-PPO) — sobre dados reais de uma rede de varejo regional brasileira (estado PB, produto 48130, 15 lojas, 38 ciclos bimestrais).

Fonte do artigo: [`docs/sbpo/main.tex`](docs/sbpo/main.tex)

---

## Pipeline de Simulação (Kedro)

Projeto Kedro completo para a dissertação de mestrado. Suporta qualquer estado ou combinação de estados/produtos com filtros configuráveis via YAML.

```bash
cd simulation/
pip install -e . -r requirements.txt

# Rodar pipeline completo (padrão: estado PB, produto 48130)
kedro run

# Todos os estados e produtos intermitentes
kedro run --params "data_ingestion.states=['all'],data_ingestion.products=null"

# Apenas validação estatística (KPIs já gerados)
kedro run --pipeline statistical_validation

# Visualizar DAG
kedro viz
```

### Pipelines

| Pipeline | Entrada | Saída |
|---|---|---|
| `data_ingestion` | Parquet particionado por UF | `scenarios.parquet`, `scenarios_meta.parquet` |
| `demand_forecasting` | Cenários | `forecasters.pkl`, `forecast_metrics.csv` |
| `inventory_simulation` | Cenários + parâmetros | `kpis.parquet` (12 políticas × lojas × replicações) |
| `statistical_validation` | KPIs | Wilcoxon, Friedman+Nemenyi, Cohen's d |
| `reporting` | KPIs + testes | Figuras PDF, tabelas LaTeX (booktabs) |

### Parâmetros configuráveis

| Arquivo | Controla |
|---|---|
| `conf/base/parameters/data_ingestion.yml` | Estados, produtos, período, CV mínimo |
| `conf/base/parameters/data_cleaning.yml` | Filtros de limpeza, thresholds de outlier |
| `conf/base/parameters/simulation.yml` | Custos, lead time, hiperparâmetros das 12 políticas |
| `conf/base/parameters/forecasting.yml` | LSTM, ANN, XGBoost |
| `conf/base/parameters/reporting.yml` | Figuras e tabelas LaTeX |

---

## 5 KPIs avaliados

| KPI | Descrição |
|---|---|
| **TIC** | Custo total de inventário (holding + ruptura + pedido) |
| **NS** | Nível de serviço — proporção da demanda atendida |
| **TR** | Taxa de ruptura — proporção de ciclos com falta |
| **BE** | Efeito Bullwhip — Var(pedidos) / Var(demanda) |
| **FP** | Frequência de pedidos — pedidos / ciclos |

---

## Requisitos

```
python >= 3.9
kedro >= 0.19
pandas, numpy, scikit-learn, xgboost
deap, scipy, scikit-posthocs
matplotlib
```

Instalação completa: `pip install -r simulation/requirements.txt`
